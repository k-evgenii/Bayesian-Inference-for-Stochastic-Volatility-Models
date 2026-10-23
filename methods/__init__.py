from pathlib import Path
import numpy as np
import os
import sys

##Storage directory

current_dir = os.getcwd()

script_dir = os.path.join(
    current_dir, "__init__.py")

data_dir =  os.path.join(
    current_dir, "data")

if __name__ == "__main__":
    print(current_dir)
    print(script_dir)
    print(data_dir)