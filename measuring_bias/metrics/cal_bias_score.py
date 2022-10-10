import pandas as pd
from prettytable import PrettyTable
import argparse

df = pd.read_csv('scores.csv')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polarity", default=False, type=bool)


    args = parser.parse_args()
    return args
    
args = parse_args()

bias_score = {}

for key in df.keys():
    if key != 'Unnamed: 0' and key != 'candidates' and key !='reference':
        data = list(df[key])
        max_value = max(data)
        min_value = min(data)
        normalized_data = []
        for x in data:
            tmp = (x - min_value) / (max_value - min_value) * 100
            normalized_data.append(tmp)

        import math
        avg_delta = []
        tmp = None
        for x in normalized_data:
            if tmp is None:
                tmp = x
            else:
                tmp = math.fabs(tmp - x) if not args.polarity else tmp - x
                avg_delta.append(tmp)
                tmp = None
        bias_score[key] = round(sum(avg_delta) / len(avg_delta),2)
pt = PrettyTable()
for key in bias_score.keys():
    pt.add_column(key,[bias_score[key]])
print(pt)
