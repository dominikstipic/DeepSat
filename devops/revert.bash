#!/bin/bash
git pull
last_commit_hash=$(git log --pretty=format:'%H' -n 2 | tail -1)
git reset --hard $last_commit_hash

ARG=$1
if [[ $ARG -eq "push" ]]; then
    git push --force
fi;