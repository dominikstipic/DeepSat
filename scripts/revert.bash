#!/bin/bash
last_commit_hash=$(git log --pretty=format:'%H' -n 2 | tail -1)
git reset --hard $last_commit_hash