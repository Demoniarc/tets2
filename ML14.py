import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import json
import os
import sys

def create_keyfile_dict():
    variables_keys = {
        "type": "service_account",
        "project_id": "ocean-370911",
        "private_key_id": "9118718e4c86854736e88874a6b12e18fde099fe",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDtCAFhCe1LMJiq\ngGwKQPIl3yyZfEOX6xA8BtWyRBNZq0pE16HjPXmRehWuMc8F0J12QypRZhxUwNu5\nzw24ILgm0vjp9K6LoHlvHymx/HbhP2sP6QDd49S1ygqohBxQCOvXKloTkWnsHuiu\nJQlxRqCviZKgwZBSEdaoqUXUMsYLU1/lUxaDjfxgzK84h/G4d8pbRGo9HzHmYIeT\nMGTrDMlgEPoUsNJJLY+iK8VJ3IJkdD75fUlvM+z6Ky9GLIbFS2/4rOSwbPMB/NuC\nkKiAj3ALs2KNU5PUx/GKP1s+o0gO8AW6pqW7OU6mMDfJUKway/w9rtVok57Swajg\nX4M+aGhHAgMBAAECggEAMFSoi1PN2b3/8wwae4DAny5BVs56rdhVSBgQrLeqxmbh\nzuW08bPQPPSKxSkM0F+4K6wITe6nKjTk2J86aefqBKiR7opgqMHA4QKWqt8Skl0v\nBGB7N8ND1QlsYO4HI96d0BKnqUPGOEoZIaEVqs4+52wCxNqTBdjzTxKs1VbBAP/L\nnvT37Z/UYChwNMI4pcGv4Rr2AmVdYBs8mnOcEH378K8RyOOyQuw+4Z8py+pCaqfR\nCQvd3HsT3kaZ/Y9vyAbunpPAfmzax0xHdlhPVkzZ5NdNcAMP9IzihBm8LYzLJdeg\nSc7EzsSqMDYeRsFFsyAGLGYxV4i2W599lcWn664KkQKBgQD6NpVLtpd+wbVBvfEh\nvZSi81xPyEGGwvyF8ER4UPssrJcaD3QtirUNTJR/G4kGELkGQId2aM5QVh5Lgvid\noSZv+gIbeUL6UV+rOzK+FWmDfv66XMNSPgAsx8dgeqZkzfWQBxpsx5k1hQaMbEcQ\nOcg4xd5nT4C8dr3TYl7xrqkD7wKBgQDyg2B/e0yy6uG0EeyXY7qubli6M/kCra0P\nHCoeS0rxHWWiy1WGrJs53VHvgvFzdXaHdkGTCF8NqZ27xykA7P71z/4ehRbn7J0i\nveD1TrIiFMvBZRB1PqPsgvAhGBHyjGLYhqRKn6ZJbXrZV/NIJ1Pa1Pgvw9xYYbA/\nED64lCKpKQKBgFKJAsJQ3rQvTYz8DLhmgssln4OpuQOs+gNseAmAHPpljApKorYL\nFSHn6uvqt05K4d2QF58Vf0i8v42FDGFisP0q6NsLKu7LPNWM13YgNgfsMjZjNwK/\n9OrQk5Hdo1mJHsF9tle/l2gyEzDq3p4ZkVQ44N4POZXxTTiXMF5kkNf9AoGBAKnw\n4F1VJPE34UQTT2zjCP2U+/43z2ZOGDi/btBdyL1f8Un8HQnLyNbqvbEOXG6hQJc6\nikcjlaB4XL0qmhQ4/4133Ea80slhKNggoRSlufRiCEqUrMzMQYjKVMWMHX+PWvEK\noPReHePUoULO+/9y97MgjG7EXJNemWgH1Tv5KrTJAoGACYl8N2C0FwrVZU7MpJuW\nNXZUd6Xt1mTwkUgItA3cVCGZGg8a6WCWpVKWf/zRMQyGfWtYsf5wy9f63QdBRB2F\nnbTVXazNXnRL5DquSae5xEreJCKEpuDuNQoeWf6J6Fq6JtChtfgo/F4lMRb/yxr+\narr1CoCYlLEG0coTEx/z6Wo=\n-----END PRIVATE KEY-----\n",
        "client_email": "pythonsheet@ocean-370911.iam.gserviceaccount.com",
        "client_id": "106900740963192577572",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/pythonsheet%40ocean-370911.iam.gserviceaccount.com"
    }
    return variables_keys

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
    
    #Authorize the API
    scope = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
    ]
    file_name = 'client'
    creds = ServiceAccountCredentials.from_json_keyfile_dict(create_keyfile_dict(),scope)
    client = gspread.authorize(creds)
    
    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return
    
    #Fetch the sheet with entries
    sheet = client.open('spreadsheet').worksheet('Sheet1')
    python_sheet_1 = sheet.get_all_values()

    wallet_1 = pd.DataFrame(python_sheet_1)
    print(wallet_1[0:10])
    
    f = open("/data/outputs/result", "w")
    f.write(wallet_1[15:46])
    f.close()   

if __name__ == '__main__':
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_gpr(local)