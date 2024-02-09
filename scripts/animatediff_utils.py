def get_animatediff_arg(p):
    """
    Get AnimateDiff argument from `p`. If it's a dict, convert it to AnimateDiffProcess.
    """
    if not p.scripts:
        return None

    for script in p.scripts.alwayson_scripts:
        if script.title().lower() == "animatediff":
            animatediff_arg = p.script_args[script.args_from]
            if isinstance(animatediff_arg, dict):
                from scripts.animatediff_ui import AnimateDiffProcess
                animatediff_arg = AnimateDiffProcess(**animatediff_arg)
                p.script_args[script.args_from] = animatediff_arg
            return animatediff_arg

    return None

def get_controlnet_units(p):
    """
    Get controlnet arguments from `p`.
    """
    if not p.scripts:
        return None

    for script in p.scripts.alwayson_scripts:
        if script.title().lower() == "controlnet":
            cn_units = p.script_args[script.args_from:script.args_to]
            return [x for x in cn_units if x.enabled]

    return None