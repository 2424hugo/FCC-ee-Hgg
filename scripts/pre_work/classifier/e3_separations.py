import awkward as ak
import numpy as np
from sklearn.metrics import roc_auc_score

sig = ak.from_parquet("cache/signal_e3_beta_02_1000.parquet")['e3_beta_0.2']
bkg = ak.from_parquet("cache/bkg_e3_beta_02_1000.parquet")['e3_beta_0.2']

sig1 = ak.from_parquet("cache/signal_1e3_beta_02_1000.parquet")['1e3_beta_02']
bkg1 = ak.from_parquet("cache/bkg_1e3_beta_02_1000.parquet")['1e3_beta_02']


sig_flat = ak.to_numpy(ak.flatten(sig, axis=None))
bkg_flat = ak.to_numpy(ak.flatten(bkg, axis=None))

sig1_flat = ak.to_numpy(ak.flatten(sig1, axis=None))
bkg1_flat = ak.to_numpy(ak.flatten(bkg1, axis=None))

y_true = np.concatenate([
    np.ones(len(sig_flat)),
    np.zeros(len(bkg_flat))
])

scores = np.concatenate([
    sig_flat,
    bkg_flat
])

print(roc_auc_score(y_true, scores))

print(np.corrcoef(sig_flat, sig1_flat)[0,1])
