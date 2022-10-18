import pandas as pd
from prettytable import PrettyTable

KEY_VALUE = {
    'bart_score_bart_base_adapter': 'bart_score_avg_f',
    'bart_score_bart_base': 'bart_score_avg_f',
    'bert_score_bert_base_adapter': 'bert_score_f',
    'bert_score_bert_base': 'bert_score_f',
    'bert_score_bert_large_adapter': 'bert_score_f',
    'bert_score_bert_large': 'bert_score_f',
    'bleurt_bert_base_adapter': 'bleurt_score',
    'bleurt_bert_base': 'bleurt_score'
}



def analyse_pkls(key,value):
    data=pd.read_pickle('pkls/' + key +'.pkl' )
    from scipy.stats import pearsonr, spearmanr, kendalltau

    human = []
    metric = []

    for i in data.keys():
        for j in data[i]['sys_summs'].keys():
            human.append(data[i]['sys_summs'][j]['scores']['litepyramid_recall'])
            metric.append(data[i]['sys_summs'][j]['scores'][value])
    correlation, p_value = spearmanr(metric, human)
    return correlation
pt = PrettyTable()

for key in KEY_VALUE.keys():
    pt.add_column(key, [analyse_pkls(key,KEY_VALUE[key])])

print(pt)