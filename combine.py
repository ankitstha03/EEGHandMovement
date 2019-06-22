import os
import pandas as pd
import glob


def combinecsv():

    os.chdir(os.path.dirname(__file__)+'/datahistory')

    extension = 'csv'
    dataset512 = pd.DataFrame()
    dataset1024 = pd.DataFrame()
    dataset2048 = pd.DataFrame()
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    for f in all_filenames:
        if ('dataset512' in f):
            dataset512 = pd.concat([dataset512, pd.read_csv(f, header=None)  ])
        elif ('dataset1024' in f):
            dataset1024 = pd.concat([dataset1024,pd.read_csv(f, header=None)])
        elif ('dataset2048' in f):
            dataset2048 = pd.concat([dataset2048, pd.read_csv(f, header=None) ])

    os.chdir(os.path.dirname(__file__)+'/datacombined')
    if os.path.exists('dataset512.csv'):
        os.remove('dataset512.csv')
    if os.path.exists('dataset1024.csv'):
        os.remove('dataset1024.csv')
    if os.path.exists('dataset2048.csv'):
        os.remove('dataset2048.csv')

    #export to csv
    dataset512.to_csv( "dataset512.csv", header=None, index=None, encoding='utf-8-sig')
    dataset1024.to_csv( "dataset1024.csv", header=None, index=None, encoding='utf-8-sig')
    dataset2048.to_csv( "dataset2048.csv", header=None, index=None, encoding='utf-8-sig')
