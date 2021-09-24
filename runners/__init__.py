import logging

import torch

LOG_NAME = "pipeline.log"
logging.basicConfig(filename=LOG_NAME, level=logging.INFO, format='%(asctime)s:%(message)s', datefmt='%d/%m/%Y %I:%M:%S')

torch.manual_seed(42)


