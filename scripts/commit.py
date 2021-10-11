import argparse
import os

from src.utils.common import current_time

DATA_TYPE = "data"
RUN_TYPE  = "run"
CODE_TYPE = "code"

########################

DVC_ADD = lambda dir: f"dvc add {dir}"
DVC_PUSH="dvc push"

GIT_COMMIT = lambda mess: f"git commit -m '{mess}'"
GIT_PUSH = "git push"

########################

def cmi_parse() -> tuple:
    parser = argparse.ArgumentParser(description="A DeepSat project versioner")
    parser.add_argument("commit_type", type=str, choices=[DATA_TYPE, RUN_TYPE, CODE_TYPE], help="the type of commit")
    parser.add_argument("--message", type=str, default="", help="A optional message for commit")
    parser.add_argument("--test", action="store_true", default=False, help="Test commit message")
    args = vars(parser.parse_args())
    return args

########################

def dvc_add(type):
    if type == RUN_TYPE:
        cmd = DVC_ADD("reports")
    else: 
        cmd = DVC_ADD("data")
    os.system(cmd)

def dvc_push():
    os.system(DVC_PUSH)

def git_add_commit(commit_type, commit_message):
    if commit_type == RUN_TYPE: 
        git_add_cmd = f"git add reports.dvc .gitignore"
    elif commit_type == DATA_TYPE: 
        git_add_cmd = f"git add data.dvc .gitignore"
    else:
        git_add_cmd = f"git add ."
    os.system(git_add_cmd)
    git_commit_cmd = GIT_COMMIT(commit_message)
    os.system(git_commit_cmd)

def git_push():
    os.system(GIT_PUSH)

######################

def form_commit_message(commit_type, timestamp, data_hash, message):
    return f"{commit_type}; {timestamp}; {data_hash}\n\n{message}"

def get_data_hash():
    hash = None
    with open("data.dvc", "r") as fp:
        for line in fp:
            if "md5" in line:
                hash = line.split(":")[1].strip()
    return hash

def process(commit_type, message, test):
    timestamp = current_time()
    data_hash = get_data_hash()
    commit_message = form_commit_message(commit_type, timestamp, data_hash, message)
    print(f"commit message: {commit_message}")
    if not test:
        dvc_add(commit_type)
        dvc_push()
        git_add_commit(commit_type, commit_message)
        git_push()

if __name__ == "__main__":
    args = cmi_parse()
    process(**args)