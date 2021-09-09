import sys
from importlib import import_module

from .compiler import config_compiler as config_compiler
from .compiler import actions as actions

def get_object_from_standard_name(std_name: str):
    pckg_parts = std_name.split(".")
    pckg_name, cls_name = ".".join(pckg_parts[:-1]), pckg_parts[-1]
    try:
        pckg = import_module(pckg_name)
        obj = getattr(pckg, cls_name)
    except ModuleNotFoundError:
        print(f"Could not find specified object in python search path: {std_name}")
        sys.exit(1)
    return obj

def import_object(obj_dict: dict, **kwargs):
    obj_std_name, params_dict = list(obj_dict.items())[0]
    
    if len(kwargs) > 0:
        locals().update(kwargs)
        eval_action = actions.eval_action_init(locals())
        params_dict = config_compiler.compile(params_dict, [eval_action])

    obj = get_object_from_standard_name(obj_std_name)
    return obj(**params_dict)