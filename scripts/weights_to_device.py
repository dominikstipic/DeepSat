import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description="Convert cuda weights to cpu weights")
parser.add_argument("--path", default="repository/trainer/output/weights.pt", help="Path to the model weights")
parser.add_argument("--device", default="cpu", help="A target device")
parser.add_argument("--out", default="weights.pt", help="name of the file")


args = parser.parse_args()
path = args.path
device = args.device
out = args.out

weights = torch.load(path, map_location=torch.device(device))
output = Path(path).parent / out
torch.save(weights, str(output))
