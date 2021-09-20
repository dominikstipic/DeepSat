#!/bin/bash
dvc remove data.dvc
dvc gc -f --workspace --cloud