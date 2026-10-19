"""Build student-cross-fitted, fixed-coordinate profiles from provenanced training events."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.data.interactions import InteractionDataset
from src.utils.config import load_config, project_path
from src.utils.files import sha256


VERSION = "dha-reference-v1"


def runtime_fingerprint(arrays):
    """Hash the exact runtime training scope, including event provenance and masks."""
    digest = hashlib.sha256()
    for name in ["studentId", "lengths", "sourceRow", "problemId", "skill", "correct"]:
        if name not in arrays:
            raise ValueError("Reference construction requires sourceRow event provenance; rebuild it from a verifiable source.")
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def extract_events(arrays):
    """Remove window overlap by (student,sourceRow), never by question/answer triples."""
    runtime_fingerprint(arrays)
    events, seen, duplicates = [], {}, 0
    for row, length in enumerate(arrays["lengths"]):
        student = int(arrays["studentId"][row])
        for position in range(int(length)):
            source = int(arrays["sourceRow"][row, position])
            skills = tuple(sorted(set(int(x) for x in np.asarray(arrays["skill"][row, position]).reshape(-1) if x > 0)))
            event = (student, source, int(arrays["problemId"][row, position]), int(arrays["correct"][row, position]), skills)
            if source < 0 or student < 1 or event[2] < 1 or event[3] not in (0, 1):
                raise ValueError("Invalid event identity or payload in training arrays.")
            key = (student, source)
            if key in seen:
                if seen[key] != event:
                    raise ValueError("Conflicting payload for the same provenanced event.")
                duplicates += 1
            else:
                seen[key] = event
                events.append(event)
    return events, duplicates


def prefix_predictions(events):
    """Predict before updating a student's Beta-smoothed global and unique-skill counts."""
    histories, probabilities = {}, []
    for student, _, _, answer, skills in events:
        correct, count, per_skill = histories.setdefault(student, [0, 0, {}])
        prior = (correct + 1.) / (count + 2.)
        values = [(per_skill.get(k, (0, 0))[0] + 5*prior) / (per_skill.get(k, (0, 0))[1] + 5) for k in skills]
        probabilities.append(float(np.mean(values)) if values else prior)
        histories[student][0] += answer
        histories[student][1] += 1
        for skill in skills:
            old_correct, old_count = per_skill.get(skill, (0, 0))
            per_skill[skill] = (old_correct + answer, old_count + 1)
    return np.asarray(probabilities, dtype=np.float64)


def _student_splits(data_dir, metadata):
    groups = {}
    for split in ["train", "valid", "test"]:
        if "split_students" in metadata:
            groups[split] = set(map(int, metadata["split_students"][split]))
        else:
            with np.load(data_dir/f"{split}.npz", allow_pickle=False) as data:
                groups[split] = set(map(int, data["studentId"]))
    if any(groups[a] & groups[b] for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]):
        raise ValueError("Training, validation and test students must be disjoint.")
    return groups


def build_reference(config, output_dir, fold_count=3, draw_count=8, bins=8, seed=42,
                    fixed_scale=4., shrinkage_kappa=5., max_windows=None, max_seq_len=None, overwrite=False):
    for name, value, minimum in [("fold_count", fold_count, 2), ("draw_count", draw_count, 1), ("bins", bins, 2)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if not math.isfinite(fixed_scale) or fixed_scale <= 0 or not math.isfinite(shrinkage_kappa) or shrinkage_kappa <= 0:
        raise ValueError("Profile scale and shrinkage must be positive and finite.")
    data_dir = project_path(config["dataset"]["paths"]["processed"])
    metadata_path = data_dir/"metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    groups = _student_splits(data_dir, metadata)
    dataset = InteractionDataset(data_dir/"train.npz", max_windows, max_seq_len)
    events, duplicate_count = extract_events(dataset.arrays)
    students = np.asarray(sorted({event[0] for event in events}), dtype=np.int64)
    if len(students) < fold_count:
        raise ValueError("Fewer training students than folds; explicitly choose at least two folds.")
    if not set(students.tolist()) <= groups["train"]:
        raise ValueError("Training runtime students are outside the saved training split.")
    output = project_path(output_dir)
    if (output/"manifest.json").exists() and not overwrite:
        raise FileExistsError("Reference artifact exists; use --overwrite to rebuild explicitly.")
    output.mkdir(parents=True, exist_ok=True)
    assignment = {int(student): index % fold_count for index, student in enumerate(np.random.default_rng(seed).permutation(students))}
    maximum_student = max(set().union(*groups.values()))
    student_fold = np.full(maximum_student+1, -1, dtype=np.int64)
    student_split = np.zeros(maximum_student+1, dtype=np.int64)
    for code, split in enumerate(["train", "valid", "test"], 1):
        student_split[list(groups[split])] = code
    for student, fold in assignment.items():
        student_fold[student] = fold
    probabilities = prefix_predictions(events)
    question = np.asarray([event[2] for event in events], dtype=np.int64)
    student = np.asarray([event[0] for event in events], dtype=np.int64)
    answers = np.asarray([event[3] for event in events], dtype=np.float64)
    bucket = np.minimum((probabilities*bins).astype(np.int64), bins-1)
    event_file = output/"prefix_predictions.npz"
    np.savez_compressed(event_file, studentId=student, sourceRow=np.asarray([e[1] for e in events]),
                        problemId=question, correct=answers.astype(np.int64), p_ref=probabilities, probability_bin=bucket)
    shape = (fold_count+1, draw_count, metadata["num_problems"]+1, bins)
    tensors = {name: np.zeros(shape, dtype=np.float32) for name in ["partial_profile", "dense_profile", "partial_counts", "dense_counts"]}
    tensors.update({name: np.zeros(shape, dtype=bool) for name in ["partial_observed", "dense_observed"]})
    views = []
    for view in range(fold_count+1):
        pool = students if view == fold_count else np.asarray([s for s in students if assignment[int(s)] != view], dtype=np.int64)
        draws = []
        for draw in range(draw_count):
            draw_seed = seed + view*100003 + draw*997
            rng = np.random.default_rng(draw_seed)
            dense_fraction = 1. if draw == 0 else float(rng.uniform(.8, 1.))
            partial_fraction = .5 if draw == 0 else float(rng.uniform(.25, .75))
            dense_n = min(len(pool), max(1, math.floor(len(pool)*dense_fraction))) if len(pool) else 0
            dense = rng.choice(pool, dense_n, replace=False)
            partial_n = min(dense_n, max(1, math.floor(dense_n*partial_fraction))) if dense_n else 0
            partial = rng.choice(dense, partial_n, replace=False)
            for name, selected in [("partial", partial), ("dense", dense)]:
                chosen = np.isin(student, selected)
                count = np.zeros(shape[-2:], dtype=np.float64)
                sums = np.zeros_like(count)
                np.add.at(count, (question[chosen], bucket[chosen]), 1)
                np.add.at(sums, (question[chosen], bucket[chosen]), answers[chosen]-probabilities[chosen])
                tensors[name+"_profile"][view, draw] = sums/(count+shrinkage_kappa)*fixed_scale
                tensors[name+"_counts"][view, draw] = count
                tensors[name+"_observed"][view, draw] = count > 0
            draws.append(dict(seed=draw_seed, dense_fraction=dense_fraction, partial_fraction=partial_fraction,
                              dense_students=sorted(map(int, dense)), partial_students=sorted(map(int, partial)),
                              degenerate_subset=bool(dense_n <= 1)))
        views.append(dict(view=view, excluded_fold=None if view == fold_count else view,
                          reference_students=sorted(map(int, pool)), draws=draws))
    supervision = sorted({(assignment[e[0]], e[2], e[4]) for e in events})
    skill_width = max(1, max(len(key[2]) for key in supervision))
    supervision_skills = np.zeros((len(supervision), skill_width), dtype=np.int64)
    for index, key in enumerate(supervision):
        supervision_skills[index, :len(key[2])] = key[2]
    tensors.update(student_fold_ids=student_fold, student_split_ids=student_split,
                   supervision_views=np.asarray([x[0] for x in supervision], dtype=np.int64),
                   supervision_problems=np.asarray([x[1] for x in supervision], dtype=np.int64),
                   supervision_skills=supervision_skills)
    tensor_file = output/"profiles.npz"
    np.savez_compressed(tensor_file, **tensors)
    events_digest = hashlib.sha256(json.dumps(events, separators=(",", ":")).encode()).hexdigest()
    manifest = dict(schema_version=1, builder_version=VERSION, dataset_metadata_sha256=sha256(metadata_path),
        train_npz_sha256=sha256(data_dir/"train.npz"), train_events_sha256=events_digest,
        train_runtime_sha256=runtime_fingerprint(dataset.arrays), split_seed=metadata.get("preprocess", {}).get("seed"),
        fold_seed=seed, fold_count=fold_count, full_train_view=fold_count, draw_count=draw_count,
        training_scope=dict(max_windows=max_windows, max_seq_len=max_seq_len, windows=len(dataset), unique_events=len(events)),
        event_identity_scheme="(studentId, sourceRow) from canonical train.npz provenance; first occurrence retains chronological order",
        deduplication_report=dict(duplicate_window_contexts_removed=duplicate_count, unique_events=len(events)),
        predictor_type="prefix_beta_skill", predictor_config=dict(beta_alpha=1., beta_beta=1., skill_prior_strength=5.),
        predictor_checkpoint_sha256=None, prefix_prediction_protocol="per-student pre-response prefix; predict then update; global once per event",
        prefix_predictions_sha256=sha256(event_file), bin_edges=np.linspace(0., 1., bins+1).tolist(),
        fixed_profile_scale=float(fixed_scale), shrinkage_kappa=float(shrinkage_kappa),
        partial_dense_sampling_protocol="student-level nested subsets; floor then min1; draw0 dense=all partial=50%",
        student_fold_assignment_sha256=hashlib.sha256(student_fold.tobytes()).hexdigest(),
        views=views, tensor_file="profiles.npz", tensor_file_sha256=sha256(tensor_file),
        profile_table_shapes={k:list(v.shape) for k,v in tensors.items()},
        canonical_skill_policy="sorted unique nonzero skills of each observed event")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output/"manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/assistment2009.yaml")
    parser.add_argument("--output-dir", help="Default: data/<dataset>/dha_kt/reference")
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--draw-count", type=int, default=8)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = args.output_dir or project_path(config["dataset"]["paths"]["processed"]).parent/"dha_kt"/"reference"
    manifest = build_reference(config, output, args.fold_count, args.draw_count, args.bins, args.seed,
                               max_windows=args.max_windows, max_seq_len=args.max_seq_len, overwrite=args.overwrite)
    print(json.dumps(dict(manifest=str(manifest), manifest_sha256=sha256(manifest)), indent=2))
    return manifest


if __name__ == "__main__":
    main()
