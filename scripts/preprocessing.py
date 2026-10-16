"""Shared static preprocessing: ID indexing, student splits and sequence packing."""
import numpy as np

FIELDS = ["studentId", "problemId", "skill", "timeTaken", "correct"]


def preprocess_dataset(config):
    """Run the dataset adapter with the training seed for reproducible student splits."""
    name = config["dataset"]["name"]
    if name == "assistment2009":
        from scripts.assistment09.preprocess import preprocess
    elif name == "assistment2017":
        from scripts.assistment17.preprocess import preprocess
    else:
        raise ValueError(f"No preprocessing adapter for {name}.")
    config["preprocess"]["seed"] = config["train"]["seed"]
    return preprocess(config)


def build_index(values):
    values = sorted(set(str(x) for x in values))
    if all(value.isdigit() for value in values):
        values.sort(key=lambda x: (int(x), x))
    return {value: index for index, value in enumerate(values, start=1)}


def split_students(student_ids, ratios, seed):
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("split_ratios must be three positive values summing to 1.")
    students = np.asarray(sorted(set(student_ids)), dtype=np.int64)
    students = np.random.default_rng(seed).permutation(students)
    ntrain, nvalid = int(len(students) * ratios[0]), int(len(students) * ratios[1])
    if min(ntrain, nvalid, len(students) - ntrain - nvalid) < 1:
        raise ValueError("Too few students: configured ratios create an empty split.")
    return dict(zip(["train", "valid", "test"],
                    [students[:ntrain], students[ntrain:ntrain+nvalid], students[ntrain+nvalid:]]))


def pack_sequences(frame, max_length, max_time, max_skills=None):
    """Overlap one context event; multi-skill events remain one prediction target."""
    if max_length < 3:
        raise ValueError("max_seq_len must be at least 3.")
    multiple = "skillIds" in frame.columns
    if multiple:
        required = int(frame.skillIds.map(len).max()) if len(frame) else 1
        max_skills = required if max_skills is None else max_skills
        if max_skills < required:
            raise ValueError("max_skills cannot truncate a real event's skills.")
    items = []
    for student, group in frame.groupby("studentId", sort=True):
        for start in range(0, len(group) - 1, max_length - 1):
            items.append((student, group.iloc[start:start+max_length]))
    count = len(items)
    integer_fields = ["problemId", "correct", "sourceRow"]
    if "orderId" in frame.columns:
        integer_fields.append("orderId")
    arrays = {name: np.zeros((count, max_length), dtype=np.int64) for name in integer_fields}
    skill_shape = (count, max_length, max_skills) if multiple else (count, max_length)
    arrays["skill"] = np.zeros(skill_shape, dtype=np.int64)
    arrays.update({name: np.zeros((count, max_length), dtype=np.float32)
                   for name in ["timeTaken", "timeNormalized"]})
    arrays["lengths"] = np.zeros(count, dtype=np.int64)
    arrays["studentId"] = np.zeros(count, dtype=np.int64)
    for row, (student, group) in enumerate(items):
        length = len(group)
        arrays["studentId"][row], arrays["lengths"][row] = student, length
        for name in integer_fields + ["timeTaken"]:
            arrays[name][row, :length] = group[name].to_numpy()
        if multiple:
            for step, skills in enumerate(group.skillIds):
                arrays["skill"][row, step, :len(skills)] = skills
        else:
            arrays["skill"][row, :length] = group.skill.to_numpy()
        arrays["timeNormalized"][row, :length] = np.log1p(group.timeTaken) / np.log1p(max_time)
    return arrays
