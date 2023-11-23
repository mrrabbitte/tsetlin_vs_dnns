
import json
import pandas as pd

if __name__ == "__main__":
    with open('grid-2023-11-20-22-36-46-656479-ANNEALING.json') as f:
        for ln in f:
            result = json.loads(ln)



            if 'dnn' in result.keys():
                print(result)
            elif 'tm' in result.keys():
                print("TM:", result)
