"""DHA-only stage policy, used through generic trainer hooks."""


def stage_for_epoch(epoch, base_warmup_epochs, diffusion_pretrain_epochs):
    if epoch <= base_warmup_epochs:
        return "base_warmup"
    if epoch <= base_warmup_epochs+diffusion_pretrain_epochs:
        return "diffusion_pretrain"
    return "joint"


def apply_stage(model, stage):
    for name, parameter in model.named_parameters():
        generator = name.startswith("diffusion.")
        delayed = name.startswith("profile_to_residual.") or name.startswith("adapter.")
        parameter.requires_grad_(stage == "joint" or stage == "base_warmup" and not (generator or delayed)
                                 or stage == "diffusion_pretrain" and generator)
