#This code snippets makes life so much easier: Using this boiler plate code I was able to get x8+ performance boot on 10 core VM.
#http://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
#https://www.linkedin.com/pulse/making-python-pandas-parallel-saurabh-sarkar-ph-d-/
 
 
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
 
def tmpFunc(df):
    df['c'] = df.a + df.b
    return df
 
def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)
 
if __name__ == '__main__':
    df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]},index= ['g1', 'g1', 'g2'])
    print 'parallel version: '
    print applyParallel(df.groupby(df.index), tmpFunc)
