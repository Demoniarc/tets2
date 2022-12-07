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
    model_0 = DecisionTreeClassifier(max_depth=depth)   
    y_0 = data[['0']].astype(np.int8)
    X_train, X_test, y_0_train, y_0_test = train_test_split(X, y_0, test_size=0.1, random_state=10)
    model_0.fit(X_train, y_0_train) 
    model_0.score(X_test, y_0_test) 
    
    model_1 = DecisionTreeClassifier(max_depth=depth)  
    y_1 = data[['1']].astype(np.int8)
    X_train, X_test, y_1_train, y_1_test = train_test_split(X, y_1, test_size=0.1, random_state=10)
    model_1.fit(X_train, y_1_train) 
    print(model_1.score(X_test, y_1_test))

    model_2 = DecisionTreeClassifier(max_depth=depth)   
    y_2 = data[['2']].astype(np.int8)
    X_train, X_test, y_2_train, y_2_test = train_test_split(X, y_2, test_size=0.1, random_state=10)
    model_2.fit(X_train, y_2_train) 
    model_2.score(X_test, y_2_test)  

    model_3 = DecisionTreeClassifier(max_depth=depth)   
    y_3 = data[['3']].astype(np.int8)
    X_train, X_test, y_3_train, y_3_test = train_test_split(X, y_3, test_size=0.1, random_state=10)
    model_3.fit(X_train, y_3_train) 
    model_3.score(X_test, y_3_test)  

    model_4 = DecisionTreeClassifier(max_depth=depth)   
    y_4 = data[['4']].astype(np.int8)
    X_train, X_test, y_4_train, y_4_test = train_test_split(X, y_4, test_size=0.1, random_state=10)
    model_4.fit(X_train, y_4_train) 
    model_4.score(X_test, y_4_test)  

    model_5 = DecisionTreeClassifier(max_depth=depth)   
    y_5 = data[['5']].astype(np.int8)
    X_train, X_test, y_5_train, y_5_test = train_test_split(X, y_5, test_size=0.1, random_state=10)
    model_5.fit(X_train, y_5_train) 
    model_5.score(X_test, y_5_test)  

    model_6 = DecisionTreeClassifier(max_depth=depth)   
    y_6 = data[['6']].astype(np.int8)
    X_train, X_test, y_6_train, y_6_test = train_test_split(X, y_6, test_size=0.1, random_state=10)
    model_6.fit(X_train, y_6_train) 
    model_6.score(X_test, y_6_test)  

    model_7 = DecisionTreeClassifier(max_depth=depth)   
    y_7 = data[['7']].astype(np.int8)
    X_train, X_test, y_7_train, y_7_test = train_test_split(X, y_7, test_size=0.1, random_state=10)
    model_7.fit(X_train, y_7_train) 
    model_7.score(X_test, y_7_test)      

    model_8 = DecisionTreeClassifier(max_depth=depth)   
    y_8 = data[['8']].astype(np.int8)
    X_train, X_test, y_8_train, y_8_test = train_test_split(X, y_8, test_size=0.1, random_state=10)
    model_8.fit(X_train, y_8_train) 
    model_8.score(X_test, y_8_test)  

    model_9 = DecisionTreeClassifier(max_depth=depth)   
    y_9 = data[['9']].astype(np.int8)
    X_train, X_test, y_9_train, y_9_test = train_test_split(X, y_9, test_size=0.1, random_state=10)
    model_9.fit(X_train, y_9_train) 
    model_9.score(X_test, y_9_test)  

    model_10 = DecisionTreeClassifier(max_depth=depth)   
    y_10 = data[['10']].astype(np.int8)
    X_train, X_test, y_10_train, y_10_test = train_test_split(X, y_10, test_size=0.1, random_state=10)
    model_10.fit(X_train, y_10_train) 
    model_10.score(X_test, y_10_test)  

    model_11 = DecisionTreeClassifier(max_depth=depth)   
    y_11 = data[['11']].astype(np.int8)    
    X_train, X_test, y_11_train, y_11_test = train_test_split(X, y_11, test_size=0.1, random_state=10)
    model_11.fit(X_train, y_11_train) 
    model_11.score(X_test, y_11_test)  

    model_12 = DecisionTreeClassifier(max_depth=depth)   
    y_12 = data[['12']].astype(np.int8)
    X_train, X_test, y_12_train, y_12_test = train_test_split(X, y_12, test_size=0.1, random_state=10)
    model_12.fit(X_train, y_12_train) 
    model_12.score(X_test, y_12_test)  

    model_13 = DecisionTreeClassifier(max_depth=depth)   
    y_13 = data[['13']].astype(np.int8)
    X_train, X_test, y_13_train, y_13_test = train_test_split(X, y_13, test_size=0.1, random_state=10)
    model_13.fit(X_train, y_13_train) 
    model_13.score(X_test, y_13_test)     

    model_14 = DecisionTreeClassifier(max_depth=depth)   
    y_14 = data[['14']].astype(np.int8)
    X_train, X_test, y_14_train, y_14_test = train_test_split(X, y_14, test_size=0.1, random_state=10)
    model_14.fit(X_train, y_14_train) 
    model_14.score(X_test, y_14_test)  

    model_15 = DecisionTreeClassifier(max_depth=depth)   
    y_15 = data[['15']].astype(np.int8)
    X_train, X_test, y_15_train, y_15_test = train_test_split(X, y_15, test_size=0.1, random_state=10)
    model_15.fit(X_train, y_15_train) 
    model_15.score(X_test, y_15_test)  

    model_16 = DecisionTreeClassifier(max_depth=depth)   
    y_16 = data[['16']].astype(np.int8)
    X_train, X_test, y_16_train, y_16_test = train_test_split(X, y_16, test_size=0.1, random_state=10)
    model_16.fit(X_train, y_16_train) 
    model_16.score(X_test, y_16_test)     

    model_17 = DecisionTreeClassifier(max_depth=depth)   
    y_17 = data[['17']].astype(np.int8)
    X_train, X_test, y_17_train, y_17_test = train_test_split(X, y_17, test_size=0.1, random_state=10)
    model_17.fit(X_train, y_17_train) 
    model_17.score(X_test, y_17_test)  

    model_18 = DecisionTreeClassifier(max_depth=depth)   
    y_18 = data[['18']].astype(np.int8)
    X_train, X_test, y_18_train, y_18_test = train_test_split(X, y_18, test_size=0.1, random_state=10)
    model_18.fit(X_train, y_18_train) 
    model_18.score(X_test, y_18_test)      

    model_19 = DecisionTreeClassifier(max_depth=depth)   
    y_19 = data[['19']].astype(np.int8)
    X_train, X_test, y_19_train, y_19_test = train_test_split(X, y_19, test_size=0.1, random_state=10)
    model_19.fit(X_train, y_19_train) 
    model_19.score(X_test, y_19_test)  

    model_20 = DecisionTreeClassifier(max_depth=depth)   
    y_20 = data[['20']].astype(np.int8)
    X_train, X_test, y_20_train, y_20_test = train_test_split(X, y_20, test_size=0.1, random_state=10)
    model_20.fit(X_train, y_20_train) 
    model_20.score(X_test, y_20_test)  

    model_21 = DecisionTreeClassifier(max_depth=depth)   
    y_21 = data[['21']].astype(np.int8)
    X_train, X_test, y_21_train, y_21_test = train_test_split(X, y_21, test_size=0.1, random_state=10)
    model_21.fit(X_train, y_21_train) 
    model_21.score(X_test, y_21_test)  

    model_22 = DecisionTreeClassifier(max_depth=depth)   
    y_22 = data[['22']].astype(np.int8)
    X_train, X_test, y_22_train, y_22_test = train_test_split(X, y_22, test_size=0.1, random_state=10)
    model_22.fit(X_train, y_22_train) 
    model_22.score(X_test, y_22_test)     

    model_23 = DecisionTreeClassifier(max_depth=depth)   
    y_23 = data[['23']].astype(np.int8)
    X_train, X_test, y_23_train, y_23_test = train_test_split(X, y_23, test_size=0.1, random_state=10)
    model_23.fit(X_train, y_23_train) 
    model_23.score(X_test, y_23_test)  

    model_24 = DecisionTreeClassifier(max_depth=depth)   
    y_24 = data[['24']].astype(np.int8)
    X_train, X_test, y_24_train, y_24_test = train_test_split(X, y_24, test_size=0.1, random_state=10)
    model_24.fit(X_train, y_24_train) 
    model_24.score(X_test, y_24_test)  

    model_25 = DecisionTreeClassifier(max_depth=depth)   
    y_25 = data[['25']].astype(np.int8)
    X_train, X_test, y_25_train, y_25_test = train_test_split(X, y_25, test_size=0.1, random_state=10)
    model_25.fit(X_train, y_25_train) 
    model_25.score(X_test, y_25_test)  

    model_26 = DecisionTreeClassifier(max_depth=depth)   
    y_26 = data[['26']].astype(np.int8)
    X_train, X_test, y_26_train, y_26_test = train_test_split(X, y_26, test_size=0.1, random_state=10)
    model_26.fit(X_train, y_26_train) 
    model_26.score(X_test, y_26_test)  

    model_27 = DecisionTreeClassifier(max_depth=depth)   
    y_27 = data[['27']].astype(np.int8)
    X_train, X_test, y_27_train, y_27_test = train_test_split(X, y_27, test_size=0.1, random_state=10)
    model_27.fit(X_train, y_27_train) 
    model_27.score(X_test, y_27_test)  

    model_28 = DecisionTreeClassifier(max_depth=depth)   
    y_28 = data[['28']].astype(np.int8)
    X_train, X_test, y_28_train, y_28_test = train_test_split(X, y_28, test_size=0.1, random_state=10)
    model_28.fit(X_train, y_28_train) 
    model_28.score(X_test, y_28_test)  

    model_29 = DecisionTreeClassifier(max_depth=depth)   
    y_29 = data[['29']].astype(np.int8)
    X_train, X_test, y_29_train, y_29_test = train_test_split(X, y_29, test_size=0.1, random_state=10)
    model_29.fit(X_train, y_29_train) 
    model_29.score(X_test, y_29_test)  

    model_30 = DecisionTreeClassifier(max_depth=depth)   
    y_30 = data[['30']].astype(np.int8)
    X_train, X_test, y_30_train, y_30_test = train_test_split(X, y_30, test_size=0.1, random_state=10)
    model_30.fit(X_train, y_30_train) 
    model_30.score(X_test, y_30_test)  

    model_31 = DecisionTreeClassifier(max_depth=depth)   
    y_31 = data[['31']].astype(np.int8)
    X_train, X_test, y_31_train, y_31_test = train_test_split(X, y_31, test_size=0.1, random_state=10)
    model_31.fit(X_train, y_31_train) 
    model_31.score(X_test, y_31_test)  

    model_32 = DecisionTreeClassifier(max_depth=depth)   
    y_32 = data[['32']].astype(np.int8)
    X_train, X_test, y_32_train, y_32_test = train_test_split(X, y_32, test_size=0.1, random_state=10)
    model_32.fit(X_train, y_32_train) 
    model_32.score(X_test, y_32_test)   

    model_33 = DecisionTreeClassifier(max_depth=depth)   
    y_33 = data[['33']].astype(np.int8)
    X_train, X_test, y_33_train, y_33_test = train_test_split(X, y_33, test_size=0.1, random_state=10)
    model_33.fit(X_train, y_33_train) 
    model_33.score(X_test, y_33_test)  

    model_34 = DecisionTreeClassifier(max_depth=depth)   
    y_34 = data[['34']].astype(np.int8)
    X_train, X_test, y_34_train, y_34_test = train_test_split(X, y_34, test_size=0.1, random_state=10)
    model_34.fit(X_train, y_34_train) 
    model_34.score(X_test, y_34_test)  

    print("Evaluating ML models")
    print("Here are the accuarcy of each model :")

    print("Add additional 80 mm jacket to hot water cylinder :",model_0.score(X_test, y_0_test))
    print("Cavity wall insulation :",model_1.score(X_test, y_0_test))
    print("Change heating to gas condensing boiler :",model_2.score(X_test, y_0_test))
    print("Change room heaters to condensing boiler :",model_3.score(X_test, y_0_test))
    print("Condensing boiler :",model_4.score(X_test, y_0_test))
    print("Condensing oil boiler with radiators :",model_5.score(X_test, y_0_test))
    print("Draughtproofing :",model_6.score(X_test, y_0_test))
    print("Flat roof insulation :",model_7.score(X_test, y_0_test))
    print("Floor insulation (solid floor) :",model_8.score(X_test, y_0_test))
    print("Floor insulation (suspended floor) :",model_9.score(X_test, y_0_test))
    print("Flue gas heat recovery device in conjunction with boiler :",model_10.score(X_test, y_0_test))
    print("Heat recovery system for mixer showers :",model_11.score(X_test, y_0_test))
    print("High heat retention storage heaters :",model_12.score(X_test, y_0_test))
    print("High heat retention storage heaters and dual immersion cylinder :",model_13.score(X_test, y_0_test))
    print("High performance external doors :",model_14.score(X_test, y_0_test))
    print("Hot water cylinder thermostat :",model_15.score(X_test, y_0_test))
    print("Increase hot water cylinder insulation :",model_16.score(X_test, y_0_test))
    print("Increase loft insulation to 270 mm :",model_17.score(X_test, y_0_test))
    print("Insulate hot water cylinder with 80 mm jacket :",model_18.score(X_test, y_0_test))
    print("Internal or external wall insulation :",model_19.score(X_test, y_0_test))
    print("Low energy lighting for all fixed outlets :",model_20.score(X_test, y_0_test))
    print("Replace boiler with biomass boiler :",model_21.score(X_test, y_0_test))
    print("Replace boiler with new condensing boiler :",model_22.score(X_test, y_0_test))
    print("Replace heating unit with condensing unit :",model_23.score(X_test, y_0_test))
    print("Replace single glazed windows with low-E double glazed windows :",model_24.score(X_test, y_0_test))
    print("Replacement glazing units :",model_25.score(X_test, y_0_test))
    print("Replacement warm air unit :",model_26.score(X_test, y_0_test))
    print("Room-in-roof insulation :",model_27.score(X_test, y_0_test))
    print("Secondary glazing to single glazed windows :",model_28.score(X_test, y_0_test))
    print("Solar photovoltaic panels, 2.5 kWp :",model_29.score(X_test, y_0_test))
    print("Solar water heating :",model_30.score(X_test, y_0_test))
    print("Time and temperature zone control :",model_31.score(X_test, y_0_test))
    print("Upgrade heating controls :",model_32.score(X_test, y_0_test))
    print("Wind turbine :",model_33.score(X_test, y_0_test))
    print("Wood pellet stove with boiler and radiators :",model_34.score(X_test, y_0_test))
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
    print("Add additional 80 mm jacket to hot water cylinder", model_0.predict(N))
    print("Probability that the model is correct :", model_0.predict_proba(N).max()*100,"%")
    print("")
    print("Cavity wall insulation", model_1.predict(N))
    print("Probability that the model is correct :", model_1.predict_proba(N).max()*100,"%")
    print("")
    print("Change heating to gas condensing boiler", model_2.predict(N))
    print("Probability that the model is correct :", model_2.predict_proba(N).max()*100,"%")
    print("")
    print("Change room heaters to condensing boiler", model_3.predict(N))
    print("Probability that the model is correct :", model_3.predict_proba(N).max()*100,"%")
    print("")
    print("Condensing boiler", model_4.predict(N))
    print("Probability that the model is correct :", model_4.predict_proba(N).max()*100,"%")    
    print("")
    print("Condensing oil boiler with radiators", model_5.predict(N))
    print("Probability that the model is correct :", model_5.predict_proba(N).max()*100,"%")
    print("")
    print("Draughtproofing", model_6.predict(N))
    print("Probability that the model is correct :", model_6.predict_proba(N).max()*100,"%")
    print("")
    print("Flat roof insulation", model_7.predict(N))
    print("Probability that the model is correct :", model_7.predict_proba(N).max()*100,"%")
    print("")
    print("Floor insulation (solid floor)", model_8.predict(N))
    print("Probability that the model is correct :", model_8.predict_proba(N).max()*100,"%")
    print("")
    print("Floor insulation (suspended floor)", model_9.predict(N))
    print("Probability that the model is correct :", model_9.predict_proba(N).max()*100,"%")
    print("")
    print("Flue gas heat recovery device in conjunction with boiler", model_10.predict(N))
    print("Probability that the model is correct :", model_10.predict_proba(N).max()*100,"%")
    print("")
    print("Heat recovery system for mixer showers", model_11.predict(N))
    print("Probability that the model is correct :", model_11.predict_proba(N).max()*100,"%")
    print("")
    print("High heat retention storage heaters", model_12.predict(N))
    print("Probability that the model is correct :", model_12.predict_proba(N).max()*100,"%")
    print("")
    print("High heat retention storage heaters and dual immersion cylinder", model_13.predict(N))
    print("Probability that the model is correct :", model_13.predict_proba(N).max()*100,"%")
    print("")
    print("High performance external doors", model_14.predict(N))
    print("Probability that the model is correct :", model_14.predict_proba(N).max()*100,"%")
    print("")
    print("Hot water cylinder thermostat", model_15.predict(N))
    print("Probability that the model is correct :", model_15.predict_proba(N).max()*100,"%")
    print("")
    print("Increase hot water cylinder insulation", model_16.predict(N))
    print("Probability that the model is correct :", model_16.predict_proba(N).max()*100,"%")
    print("")
    print("Increase loft insulation to 270 mm", model_17.predict(N))
    print("Probability that the model is correct :", model_17.predict_proba(N).max()*100,"%")
    print("")
    print("Insulate hot water cylinder with 80 mm jacket", model_18.predict(N))
    print("Probability that the model is correct :", model_18.predict_proba(N).max()*100,"%")
    print("")
    print("Internal or external wall insulation", model_19.predict(N))
    print("Probability that the model is correct :", model_19.predict_proba(N).max()*100,"%")
    print("")
    print("Low energy lighting for all fixed outlets", model_20.predict(N))
    print("Probability that the model is correct :", model_20.predict_proba(N).max()*100,"%")
    print("")
    print("Replace boiler with biomass boiler", model_21.predict(N))
    print("Probability that the model is correct :", model_21.predict_proba(N).max()*100,"%")
    print("")
    print("Replace boiler with new condensing boiler", model_22.predict(N))
    print("Probability that the model is correct :", model_22.predict_proba(N).max()*100,"%")
    print("")
    print("Replace heating unit with condensing unit", model_23.predict(N))
    print("Probability that the model is correct :", model_23.predict_proba(N).max()*100,"%")
    print("")
    print("Replace single glazed windows with low-E double glazed windows", model_24.predict(N))
    print("Probability that the model is correct :", model_24.predict_proba(N).max()*100,"%")
    print("")
    print("Replacement glazing units", model_25.predict(N))
    print("Probability that the model is correct :", model_25.predict_proba(N).max()*100,"%")    
    print("")
    print("Replacement warm air unit", model_26.predict(N))
    print("Probability that the model is correct :", model_26.predict_proba(N).max()*100,"%")
    print("")
    print("Room-in-roof insulation", model_27.predict(N))
    print("Probability that the model is correct :", model_27.predict_proba(N).max()*100,"%")    
    print("")
    print("Secondary glazing to single glazed windows", model_28.predict(N))    
    print("Probability that the model is correct :", model_28.predict_proba(N).max()*100,"%")
    print("")
    print("Solar photovoltaic panels, 2.5 kWp", model_29.predict(N))
    print("Probability that the model is correct :", model_29.predict_proba(N).max()*100,"%")
    print("")
    print("Solar water heating", model_30.predict(N))
    print("Probability that the model is correct :", model_30.predict_proba(N).max()*100,"%")
    print("")
    print("Time and temperature zone control", model_31.predict(N))
    print("Probability that the model is correct :", model_31.predict_proba(N).max()*100,"%")
    print("")
    print("Upgrade heating controls", model_32.predict(N))
    print("Probability that the model is correct :", model_32.predict_proba(N).max()*100,"%")
    print("")
    print("Wind turbine", model_33.predict(N))
    print("Probability that the model is correct :", model_33.predict_proba(N).max()*100,"%")
    print("")
    print("Wood pellet stove with boiler and radiators", model_34.predict(N))
    print("Probability that the model is correct :", model_34.predict_proba(N).max()*100,"%")
    print("")




    f = open("/data/outputs/result", "w")
    f.write(str(model_1.score(X_test, y_1_test)))
    f.close()    


if __name__ == '__main__':
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_gpr(local)