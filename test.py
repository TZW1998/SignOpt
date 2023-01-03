import argparse
from time import sleep
import random

parser = argparse.ArgumentParser(description='Process the hyperparameter.')
parser.add_argument("--gpu")
parser.add_argument("--run_name")
parser.add_argument("--rerun", type = bool)
parser.add_argument("--rep", type = int)

parser.add_argument("--epoch")
parser.add_argument("--bz")

args = parser.parse_args()


if __name__ == "__main__":
    sleep(10)
    
    assert random.randint(0,10) >= 2
