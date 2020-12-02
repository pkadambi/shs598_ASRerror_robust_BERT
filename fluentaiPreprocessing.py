import pandas as pd
import numpy as np
import csv

def generate_and_preprocess_fluentai(corrupted_path):
    '''
    fixes the issue wiht out of place space character added before punctuations by the ASR error simulator
    :param filepath:
    :param corrupted_path:
    :return:
    '''
    replacement_data = []
    to_replace = [[' n\'t', 'n\'t'],  [' \'d','\'d'], [' \'s', '\'s'], [' \'ll', '\'ll'], [' \'ve', '\'ve'], [' \'re', '\'re'], [' \'m', '\'m'],
                    [' ?', '?'], [' .','.'], [' !','!'], [' ,',','], [']', '']]
    with open(corrupted_path) as sentences:
        csvfile = csv.reader(sentences, delimiter=',')

        for sen_corrupted in csvfile:
            for pair in to_replace:
                if len(sen_corrupted)>1:
                    sen_corrupted = [','.join(sen_corrupted)]
                if len(sen_corrupted)==0:
                    break
                sen_corrupted[0] = sen_corrupted[0].replace(pair[0], pair[1])
            if len(sen_corrupted)==0:
                continue
            else:
                replacement_data.append(sen_corrupted)

            # print('\t'.join(row1))
            # replacement_tsv_data.append(['\t'.join(row1))

    with open(corrupted_path, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(replacement_data)


def generate_fluentai_labels(filepath, save_csv=False):
    '''
    Maps each label to a 31 class classification problem
    :return:
    '''
    df = pd.read_csv(filepath)
    df['total action'] = df['action'].str.cat(df['object'], sep='').str.cat(df['location'], sep='')
    labelDictionary = {}
    i=0
    labels = []

    for action in df['total action']:
        action_ = str(action)
        if action_ not in labelDictionary.keys():
            labelDictionary[action_] = i
            i+=1
        labels.append(labelDictionary[action_])

    df['label'] = labels
    if save_csv:
        df.to_csv(filepath)
    return df

_SAVE_CSV=True #Set this to true first time setting up the dataset


# add label corresponding to 31 unique intents
df = generate_fluentai_labels('./data/fluentai/test_data.csv', save_csv=_SAVE_CSV)
df = generate_fluentai_labels('./data/fluentai/train_data.csv', save_csv=_SAVE_CSV)
df = generate_fluentai_labels('./data/fluentai/valid_data.csv', save_csv=_SAVE_CSV)


# Remove unnecessary spaces from fluentai corrupted
fluentaipath = './data/fluentai'
noisedDataPath = './data/fluentaiCorrupted%0.2f/'
fluentaiNoiseLevels = [0.05, 0.10, 0.15, 0.2, 0.25, 0.3]
datasplits = ['test', 'train', 'valid']

for noise_level in fluentaiNoiseLevels:
    datapath = noisedDataPath % noise_level
    for split in datasplits:
        corrupted_sentence = datapath + 'corrupted_%s_utterances.csv' % split
        generate_and_preprocess_fluentai(corrupted_sentence)

