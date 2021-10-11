#!/bin/bash
dvc remove reports.dvc
dvc gc -f --workspace --cloud