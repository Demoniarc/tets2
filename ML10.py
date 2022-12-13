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

        return "data11.csv"

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

    #data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.int8)
    data0 = pd.read_csv(filename, sep=',', dtype=np.int8)
    data = data0.iloc[1:,:]

    print("Stacking data.")

    X = data.drop(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34'], axis=1)
    X = X.to_numpy()

    depth = 10
    
    print("Building ML models")
    
    model_1 = DecisionTreeClassifier(max_depth=depth)  
    y_1 = data[['1']].astype(np.int8)
    X_train, X_test, y_1_train, y_1_test = train_test_split(X, y_1, test_size=0.1, random_state=10)
    model_1.fit(X_train, y_1_train) 

    print("Evaluating ML models")
    print("Here are the accuarcy of each model :")

    print("Cavity wall insulation :",model_1.score(X_test, y_1_test))
    print("")
   
    print("Forecast")

    #Need a second dataset here with the user's inputs
    #Let's use this array as an example :
    inputs = data0.loc[0:0,:]
    N = inputs.iloc[:,0:3133].to_numpy().astype(np.uint8)

    #print(N)

    print("")
    print("Here are the improvements the model advises you to make : ")
    print("")
    print("Cavity wall insulation", model_1.predict(N))
    print("Probability that the model is correct :", model_1.predict_proba(N).max()*100,"%")
    print("")

    f = open("/data/outputs/result", "w")
    f.write(str(model_1.score(X_test, y_1_test)))
    f.close()    

if __name__ == '__main__':
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_gpr(local)