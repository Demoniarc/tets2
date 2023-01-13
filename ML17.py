import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import json
import os
import sys

def get_input(local=False):
    if local:
        print("Reading local file")

        return "tkt.csv"

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
    
    print("test")
    
    f = open("/data/outputs/result", "w")
    f.write('test')
    f.close()   

if __name__ == '__main__':
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_gpr(local)
