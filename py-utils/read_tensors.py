import safetensors
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

with safetensors.safe_open(args.filename, framework="np") as f:
    print(f"tensors: {f.keys()}")
    for key in f.keys():
        print(f"tensor {key}:")
        print(f.get_tensor(key))