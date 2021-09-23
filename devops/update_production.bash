#!/bin/bash

PATTERN="(__pycache__)|./.git/*|.*\.pytest_cache.*|./repository/*|./tests/*|./data/*|./.dvc/*"
files=($(find . -type f | egrep -v $PATTERN))

git clone "git@github.com:dominikstipic/DeepSat-production.git"
for file in "${files[@]}"
do
    FILE_ROOT=$(echo $file | cut -d/ -f2-)
    TARGET="DeepSat-production/$FILE_ROOT"
    echo $TARGET
	cp -f $file $TARGET

done

cd DeepSat-production
HASH="f0da8bb651b915982e2ab70e13e32269226ee924"
python -m devops.commit code --message "merged $HASH into production"
cd ..
rm -rf DeepSat-production
