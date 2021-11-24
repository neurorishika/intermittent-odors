from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import os
import zipfile
import shutil

def to_significance_label(p):
    if 1.00e-02 < p and p <= 5.00e-02:
        return "*"
    elif 1.00e-03 < p and p <= 1.00e-02:
        return "**"
    elif 1.00e-04 < p and p <= 1.00e-03:
        return "***"
    elif p <= 1.00e-04:
        return "****"
    else:
        return "ns"

def rankbiserial(x,y,resamples=5000,ci=0.95):
    x = x.flatten()
    y = y.flatten()
    idxs = np.vstack([np.arange(x.shape[0]),np.random.choice(x.shape[0],size=(resamples,x.shape[0]),replace=True)])
    corrs = []
    for idx in idxs:
        d = x[idx]-y[idx]
        r = stats.rankdata(abs(d))
        r_plus = np.sum((d > 0) * r)
        r_minus = np.sum((d < 0) * r)
        p_plus = r_plus/(r_plus+r_minus)
        p_minus = r_minus/(r_plus+r_minus)
        corrs.append(p_plus-p_minus)
    if ci=='sd':
        return corrs[0],np.std(corrs[1:])
    else:
        return corrs[0],(np.quantile(corrs[1:],1-ci),np.quantile(corrs[1:],ci))
    
def cliffdelta(x,y,resamples=5000,ci=0.95):
    x = x.flatten()
    y = y.flatten()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    idxxs = np.vstack([np.arange(x.shape[0]),np.random.choice(x.shape[0],
                                                              size=(resamples,x.shape[0]),replace=True)])
    idxys = np.vstack([np.arange(y.shape[0]),np.random.choice(y.shape[0],
                                                              size=(resamples,y.shape[0]),replace=True)])
    δs = []
    for i in range(resamples+1):
        U, _ = stats.mannwhitneyu(x[idxxs[i]], y[idxys[i]], alternative='two-sided')
        cliffs_delta = ((2 * U) / (len(x) * len(y))) - 1
        δs.append(cliffs_delta)
    if ci=='sd':
        return δs[0],np.std(δs[1:])
    else:
        return δs[0],(np.quantile(δs[1:],1-ci),np.quantile(δs[1:],ci))

def clean_data_cache():
    for i in os.listdir('__datacache__/'):
        os.remove("__datacache__/"+i)

def fetch_data(switch_prob):
    switch_probx10=int(switch_prob*10)
    with zipfile.ZipFile(f'../data/zip_0_{switch_probx10}.zip', 'r') as zip_ref:
        for member in tqdm(zip_ref.namelist(), desc='Extracting '):
            try:
                filename = os.path.basename(member)
                if not filename:
                    continue
                source = zip_ref.open(member)
                target = open(os.path.join("__datacache__/", filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
            except zipfile.error as e:
                pass