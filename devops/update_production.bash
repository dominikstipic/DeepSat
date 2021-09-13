#!/bin/bash

DATA_VERSION=$(git log -1 --pretty=%B | head -1 | cut -d " " -f 2)
REPORT_VERSION=$(git log -1 --pretty=%B | head -2 | tail -1 | cut -d " " -f 2)

git checkout master
MASTER_HASH=$(git log --pretty=format:'%H' -n 1 | tail -1 )
git checkout production

git merge master -m \
"data $DATA_VERSION 
report $REPORT_VERSION

merged master $MASTER_HASH into production
"
