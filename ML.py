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
     

def run_gpr(local=False):
    
    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return

    data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.int8)

    print("Stacking data.")
    X = data[:,0:3133]
    y_1 = data[:,3133:3134]
    
    print("Building ML model")
    model_1 = DecisionTreeClassifier(max_depth=10)  
    X_train, X_test, y_1_train, y_1_test = train_test_split(X, y_1, test_size=0.1, random_state=10)
    model_1.fit(X_train, y_1_train) 
    print(model_1.score(X_test, y_1_test))

    f = open("/data/outputs/result", "w")
    f.write(str(model_1.score(X_test, y_1_test)))
    f.close()    


if __name__ == '__main__':
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_gpr(local)


