#!/bin/bash
DATA_VERSION=$(git log -1 --pretty=%B | head -1 | cut -d " " -f 2)
REPORT_VERSION=$(git log -1 --pretty=%B | head -2 | tail -1 | cut -d " " -f 2)
echo old_version: data=$DATA_VERSION, report=$REPORT_VERSION
DATA_VERSION=$((DATA_VERSION + 1))
REPORT_VERSION=$((REPORT_VERSION))

dvc add data
dvc push

git add data.dvc .gitignore 
git commit -m "data $DATA_VERSION 
report $REPORT_VERSION"

echo new_version: data=$DATA_VERSION, report=$REPORT_VERSION
