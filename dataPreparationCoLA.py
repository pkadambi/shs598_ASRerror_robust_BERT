import csv
'''

This script performs data cleaning. The ASR error simulator introduces spaces erroneously into the corpus
Do not run this twice! running it on an already cleaned dataset will corrupt the file.

'''


def generate_and_preprocess_data(filepath, corrupted_path):
    '''
    Removes the extra space added by the ASR error generator from the sentences in the dataset
    :param filepath:
    :param corrupted_path:
    :return:
    '''
    delimiter = '\t'

    replacement_data = []
    to_replace = [[' n\'t', 'n\'t'],  [' \'d','\'d'], [' \'s', '\'s'], [' \'ll', '\'ll'], [' \'ve', '\'ve'], [' \'re', '\'re'], [' \'m', '\'m'],
                    [' ?', '?'], [' .','.'], [' !','!'], [' ,',','], [']', '']]
    with open(filepath) as datafile, open(corrupted_path) as sentences:
        tsv1 = csv.reader(datafile, delimiter='\t')
        tsv2 = csv.reader(sentences, delimiter='\t')

        for row1, sen_corrupted in zip(tsv1, tsv2):
            for pair in to_replace:
                sen_corrupted[0] = sen_corrupted[0].replace(pair[0], pair[1])
            row1[3] = sen_corrupted[0]

            # print('\t'.join(row1))
            # replacement_tsv_data.append(['\t'.join(row1))
            replacement_data.append(row1)

    with open(filepath, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(replacement_data)
    print()


#TODO: make this one function
'''
Preprocess to remove unncessary spaces from the CoLA dataset
'''
colaDataPath = './data/CoLA'
noisedDataPath = './data/CoLACorrupted%0.2f/'
colaNoiseLevels = [0.05, 0.1, 0.15,0.2, 0.25, 0.3]
datasplits = ['dev', 'train']
for noise_level in colaNoiseLevels:
    datapath = noisedDataPath % noise_level
    for split in datasplits:
        corrupted_sentence = datapath + 'corrupted_%s_sentences.tsv' % split
        tsvpath = datapath + '%s.tsv' % split
        generate_and_preprocess_data(tsvpath, corrupted_sentence)
