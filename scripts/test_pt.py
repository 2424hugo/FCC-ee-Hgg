import awkward as ak
import numpy as np

sig_pt = ak.from_parquet("cache/signal_jet_transverseMomentum.parquet")
sig_pt = sig_pt['jet_transverseMomentum']


print(len(sig_pt))
print(sig_pt[0])
