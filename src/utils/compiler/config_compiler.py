import copy
"""
    Module which contains the logic about config compiler functionality. 
    Config compiler takes the json configuration file and rewrites it to the appropriate format. The motivation for developing compiler is that 
    our configuration language can have special keywords which have some specific meaning. After analyzing file compiler notices this special keywords
    and changes them to some string representation with whome we can work. 
"""

def _is_recursive(value): 
  if value == None: 
    return False
  if type(value) in [str, int, float, bool]:
    return False
  else: 
    return True

def compile(config: dict, actions: list) -> dict:
  def parse_config_inner(stage_dict):
    if not _is_recursive(stage_dict):
      for action in actions: 
        if type(stage_dict) == str:
          stage_dict = action(stage_dict, config)
      return stage_dict
    iter_obj = stage_dict if type(stage_dict) == list else stage_dict.items()
    if type(stage_dict) == dict:
      for stage_name, stage_value in iter_obj:
        stage_dict[stage_name] = parse_config_inner(stage_value)
    else:
      for k in range(len(iter_obj)):
        iter_obj[k] = parse_config_inner(iter_obj[k])
    return stage_dict
  config_copy = copy.deepcopy(config)
  return parse_config_inner(config_copy)