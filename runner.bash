#!/bin/bash

# Set Python path
export PYTHONPATH="$(pwd)"
rm -r repository/*
###############################

python -m runners.preprocess
python -m runners.sharding
python -m runners.data_stat
python -m runners.dataset_factory
python -m runners.trainer
python -m runners.evaluation