import pandas as pd
import numpy as np          
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier
import json
import os
import sys

def get_input(local=False):
    if local:
        print("Reading local file")

        return "data.csv"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename

if __name__ == '__main__':
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    get_input(local)

