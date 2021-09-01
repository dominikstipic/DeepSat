import copy

def _query_dict(dictionary, query):
  copy_dict = copy.deepcopy(dictionary)
  for part in query.split("."):
    copy_dict = copy_dict[part]
  return copy_dict

def reference_action(stage_dict, config):
  parts = stage_dict.split("$")
  if len(parts) == 3 and parts[0] == "" and parts[-1] == "":
    query_string = parts[1]
    return _query_dict(config, query_string)
  return stage_dict

def eval_action_init(context):
  def eval_action(stage_dict, *args):
    locals().update(context)
    parts = stage_dict.split("%")
    if len(parts) == 3 and parts[0] == "" and parts[-1] == "":
      code = parts[1]
      return eval(code)
    return stage_dict
  return eval_action
  