# FCC-ee H → gg Event Classification

Event-level discrimination of resonant

[
e^+e^- \rightarrow H \rightarrow gg
]

from the dominant continuum hadronic background

[
e^+e^- \rightarrow q\bar q
]

at the FCC-ee Higgs pole.

This repository contains the data-processing, jet-substructure, machine-learning and statistical-analysis workflow developed for an MSc dissertation in Nuclear and Particle Physics at the University of Edinburgh.

The broader physics motivation is the possibility of constraining the electron Yukawa coupling through direct resonant Higgs production,

[
e^+e^- \rightarrow H,
]

at (\sqrt{s}\simeq m_H). The (H\rightarrow gg) final state is particularly interesting because the Higgs branching fraction to gluons is relatively large while the principal experimental challenge is rejection of the enormous (e^+e^-\rightarrow q\bar q) continuum.

---

## Analysis overview

The analysis uses simulated FCC-ee events at approximately

[
\sqrt{s}=125~\mathrm{GeV},
]

with detector response modelled using the IDEA Delphes configuration.

### Signal

```text
e+ e- → H → gg
```

### Background

```text
e+ e- → q qbar
```

The analysis proceeds through:

1. ROOT/EDM4hep event processing;
2. event and jet preselection;
3. reconstruction of the two highest-energy jets;
4. jet-constituent extraction;
5. calculation of high-level kinematic and jet-substructure observables;
6. training of several machine-learning classifiers;
7. evaluation on a held-out test sample;
8. conversion of classifier efficiencies to physical signal and background yields.

The final analysis primarily uses an event-level fully connected neural network. A boosted decision tree and ParticleNet-style constituent network are retained as comparison models.

---

# Physics motivation

Direct Higgs production at the FCC-ee provides a possible probe of the electron Yukawa coupling,

[
y_e = \frac{\sqrt{2}m_e}{v}.
]

For a monochromatised FCC-ee run near the Higgs resonance, previous studies estimate a direct Higgs production cross section of approximately

[
\sigma(e^+e^-\rightarrow H)\approx 0.28~\mathrm{fb},
]

for a centre-of-mass energy spread comparable to the Higgs width.

Using

[
\mathrm{BR}(H\rightarrow gg)\approx 8.2%,
]

gives the benchmark signal cross section used in this analysis,

[
\sigma(H\rightarrow gg)\approx 0.023~\mathrm{fb}.
]

The corresponding continuum hadronic background is approximately

[
\sigma(e^+e^-\rightarrow q\bar q)\approx 61~\mathrm{pb}.
]

The signal-to-background hierarchy is therefore extremely severe, making background rejection rather than raw classification accuracy the central challenge.

The benchmark integrated luminosity used for the final physics study is

[
\mathcal{L}=10~\mathrm{ab}^{-1}.
]

---

# Dataset and event selection

The analysis is based on FCC-ee Winter 2023 simulated samples using the IDEA detector configuration.

The processed dataset is divided at the source-file level into independent

```text
train/
validation/
test/
```

splits, preventing events originating from the same ROOT file from leaking between datasets.

The main event selection requires

[
m_{\mathrm{event}}>120~\mathrm{GeV}.
]

For each retained event, the two highest-energy reconstructed jets are selected and ordered as

```text
leading jet
subleading jet
```

Jet constituents are reconstructed through the EDM4hep jet-particle relations and stored together with their four-vectors and particle information.

The resulting Parquet dataset has the form

```text
cache/analysis_dataset/
├── signal/
│   ├── train/
│   ├── validation/
│   └── test/
└── background/
    ├── train/
    ├── validation/
    └── test/
```

Large intermediate datasets are intentionally excluded from version control.

---

# High-level observables

The event-level classifiers use reconstructed event, jet and jet-substructure information.

Important quantities include:

### Event-level quantities

* event invariant mass;
* original jet multiplicity.

### Jet kinematics

For the leading and subleading jets:

* energy;
* mass;
* momentum (p);
* transverse momentum (p_T);
* polar angle (\theta);
* constituent multiplicity.

### Energy correlation functions

Jet substructure is characterised using energy-correlation functions with

[
\beta=0.2.
]

For constituent energy fractions

[
z_i=\frac{E_i}{E_J},
]

the two-point energy correlation function is

[
e_2^{(\beta)}
=============

\sum_{i<j}
z_i z_j \theta_{ij}^{\beta},
]

and the three-point function is

[
e_3^{(\beta)}
=============

\sum_{i<j<k}
z_i z_j z_k
\left(
\theta_{ij}\theta_{ik}\theta_{jk}
\right)^{\beta}.
]

The derived observables

[
C_2=\frac{e_3}{e_2^2},
\qquad
D_2=\frac{e_3}{e_2^3}
]

are also included.

These observables provide sensitivity to differences in the radiation structure of quark and gluon jets.

---

# Machine-learning models

Several classifiers were investigated.

## Event-level neural network

The final classifier is a multilayer perceptron acting on the engineered event-level feature vector.

The frozen final architecture is

```text
Input
  ↓
256
  ↓
128
  ↓
64
  ↓
Output
```

with:

```text
activation     = ReLU
dropout        = 0.15
batch norm     = enabled
optimiser      = AdamW
learning rate  = 1e-3
weight decay   = 1e-4
```

The architecture was selected using validation data and multi-seed studies before the final refit.

The final network is then refitted using all available training and validation events with the architecture and training procedure frozen.

The test sample is not used during this final training stage.

---

## Boosted decision tree

A `HistGradientBoostingClassifier` using the same engineered event-level information is retained as a conventional machine-learning benchmark.

This provides a useful comparison between tree-based classification and the nonlinear representation learned by the neural network.

---

## ParticleNet

A ParticleNet-inspired graph neural network was also developed using reconstructed jet constituents.

Both event-level and single-jet variants were investigated.

ParticleNet operates directly on constituent-level information rather than exclusively on engineered observables, providing a useful test of whether additional discrimination can be extracted from the internal particle structure of the jets.

---

# Final test-set performance

All quoted final NN and BDT metrics below use the same held-out test sample:

```text
Signal events:      185,017
Background events:  391,553
Total:              576,570
```

The final model comparison is:

| Model         |     ROC AUC | Average precision |
| ------------- | ----------: | ----------------: |
| Final wide NN | **0.94284** |       **0.88011** |
| BDT           |     0.93641 |           0.86847 |
| ParticleNet   |     0.90071 |          0.88426* |

*ParticleNet was evaluated on a smaller balanced 10,000-event test sample and is therefore not directly identical in evaluation conditions to the full NN/BDT test.

The event-level NN gives the best overall ROC performance.

---

## Background rejection

For the final wide neural network:

| Signal efficiency | Background efficiency | Background rejection |
| ----------------: | --------------------: | -------------------: |
|               50% |                0.0211 |                 47.4 |
|               70% |                0.0541 |                 18.5 |
|               80% |                0.0881 |                 11.3 |
|               90% |                0.1608 |                 6.22 |

The BDT gives, for comparison:

| Signal efficiency | Background efficiency | Background rejection |
| ----------------: | --------------------: | -------------------: |
|               50% |                0.0242 |                 41.4 |
|               70% |                0.0607 |                 16.5 |
|               80% |                0.0981 |                 10.2 |
|               90% |                0.1781 |                 5.62 |

The neural network therefore provides a consistent improvement over the BDT across the tested working points.

---

# Feature-dependence study

A leave-one-feature-out ablation study was performed to investigate which observables carry the greatest discrimination information.

The largest reductions in validation AUC were obtained when removing:

1. leading-jet (C_2^{(\beta=0.2)});
2. subleading-jet (C_2^{(\beta=0.2)});
3. subleading-jet polar angle;
4. leading-jet polar angle;
5. event invariant mass;
6. leading-jet constituent multiplicity.

This demonstrates that the neural network is using a combination of jet-substructure and event-level topology rather than relying on a single dominant observable.

---

# Physical event rates

The final classifier is evaluated using the benchmark

```text
signal cross section      = 0.023 fb
background cross section  = 61,000 fb
integrated luminosity     = 10,000 fb^-1
```

with generator-level normalisation based on

```text
signal test generation      = 200,000 events
background test generation  = 1,200,000 events
```

The (m_{\mathrm{event}}>120) GeV preselection gives approximately

```text
signal efficiency      = 0.9251
background efficiency  = 0.3263
```

corresponding to expected pre-NN yields of approximately

[
S \approx 213,
]

and

[
B \approx 1.99\times10^8
]

for (10~\mathrm{ab}^{-1}).

This illustrates an important result of the study:

> A high ROC AUC does not by itself imply experimental sensitivity when the physical background cross section exceeds the signal by many orders of magnitude.

---

## Example NN operating point

With statistical uncertainty only, the selected operating point is approximately

```text
NN threshold             = 0.9419
signal efficiency        = 0.3708
background efficiency    = 0.0110
```

giving expected yields

[
S \approx 78.9,
\qquad
B \approx 2.18\times10^6,
]

and

[
Z_A \approx 0.053.
]

The significance decreases further once background systematic uncertainties are included.

The principal limitation of the analysis is therefore the extreme physical rate difference between (H\rightarrow gg) and continuum (q\bar q) production rather than insufficient classifier ROC performance.

---

# Repository structure

```text
FCC-ee-Hgg/
├── README.md
├── setup.sh
├── config.py
├── config/
│
├── scripts/
│   ├── data_processing/
│   │   ├── add_hl_observables.py
│   │   ├── enrich_analysis_shards.py
│   │   └── ...
│   │
│   ├── data_checks/
│   │   └── ...
│   │
│   ├── plotting/
│   │   └── ...
│   │
│   ├── ML/
│   │   ├── train_bdt_22_variables.py
│   │   ├── train_event_nn_architecture_sweep.py
│   │   ├── run_nn_multiseed_sweep.sh
│   │   ├── summarise_nn_multiseed.py
│   │   ├── run_nn_feature_ablation.py
│   │   ├── plot_nn_feature_dependence.py
│   │   ├── train_final_wide_all_data.py
│   │   ├── evaluate_bdt_full_test.py
│   │   ├── compare_final_models.py
│   │   ├── scan_nn_luminosity_significance.py
│   │   ├── single_jet_particlenet.py
│   │   └── ...
│   │
│   └── pre_work/
│
├── outputs/
│   ├── ml/
│   │   ├── bdt_22_variables/
│   │   ├── bdt_22_variables_test/
│   │   ├── nn_architecture_sweep_100k/
│   │   ├── nn_multiseed_sweep/
│   │   ├── nn_feature_dependence/
│   │   ├── final_feature_dependence/
│   │   ├── nn_final_wide_all_data/
│   │   ├── nn_final_wide_all_data_test/
│   │   └── final_model_comparison/
│   └── plots/
│
├── results/
│   ├── particlenet_optimised_10000/
│   └── single_jet_particlenet_v3_10000/
│
└── archive/
```

`archive/` contains superseded or exploratory analysis material retained for provenance.

---

# Environment

On CERN LXPLUS, initialise the environment from the repository root with

```bash
source setup.sh
```

The project primarily uses:

```text
Python
NumPy
Awkward Array
Uproot
PyArrow
Pandas
Matplotlib
scikit-learn
PyTorch
PyTorch Geometric
```

CUDA acceleration is used for the neural-network and ParticleNet studies where available.

---

# Typical workflow

## 1. Build the analysis dataset

The ROOT input data are converted into Parquet shards using the scripts under

```text
scripts/data_processing/
```

The processing stage resolves reconstructed jet constituents and applies the event-level selection.

---

## 2. Add jet-substructure observables

For example:

```bash
python -m scripts.data_processing.enrich_analysis_shards \
    --all \
    --dataset-root cache/analysis_dataset/signal/train \
    --chunk-size 500
```

The process should be repeated for the required signal/background splits.

---

## 3. Validate processed data

Run the validation utilities in

```text
scripts/data_checks/
```

before training models.

---

## 4. Train baseline BDT

```bash
python -m scripts.ML.train_bdt_22_variables \
    --dataset-root cache/analysis_dataset \
    --max-events-per-class 100000 \
    --importance-events 20000
```

---

## 5. Neural-network architecture study

```bash
python -m scripts.ML.train_event_nn_architecture_sweep
```

The selected network architecture was subsequently tested across multiple random seeds before being frozen.

---

## 6. Final neural-network refit

```bash
python -m scripts.ML.train_final_wide_all_data
```

This trains the frozen

```text
[256, 128, 64]
```

network on the combined training and validation datasets.

---

## 7. Final test evaluation

The held-out test split is evaluated separately.

Results are written to

```text
outputs/ml/nn_final_wide_all_data_test/
```

including:

```text
test_metrics.json
test_predictions.csv
test_roc.png
test_score_distribution.png
threshold_scan.csv
best_operating_points.csv
weighted_significance_vs_threshold.png
```

---

## 8. Compare models

```bash
python -m scripts.ML.compare_final_models
```

Outputs include:

```text
model_comparison_metrics.csv
model_comparison_operating_points.csv
model_comparison_roc.png
model_comparison_roc_low_fpr.png
```

---

# Interpretation

The study separates two distinct questions.

### Machine-learning question

Can reconstructed event and jet-substructure information distinguish (H\rightarrow gg) from (q\bar q)?

**Yes.**

The final event-level network reaches

[
\mathrm{AUC}=0.9428
]

and improves background rejection relative to the BDT.

### Physics-sensitivity question

Is this rejection sufficient to isolate direct (H\rightarrow gg) production at realistic FCC-ee cross sections?

**Not with the current analysis.**

The background production rate is so large that even strong machine-learning discrimination leaves a background yield many orders of magnitude larger than the signal.

The study therefore illustrates the distinction between classifier performance and achievable collider sensitivity.

---

# Reproducibility notes

* Train, validation and test samples are separated at source-file level.
* The test sample is excluded from model training.
* Architecture selection was performed using validation data.
* The final network architecture and hyperparameters were frozen before the final refit.
* The final refit combines training and validation data.
* Because the test set had been inspected during earlier development, the final procedure should be described as a **frozen final analysis/refit**, rather than as a completely untouched blind test.
* Physics significance calculations use cross-section and luminosity weights rather than class-balanced ML weights.
* Very-low-background operating points are restricted by the finite Monte Carlo statistics available in the background test sample.

---

# References

1. D. d'Enterria, A. Poldaru and G. Wojcik,
   *Measuring the electron Yukawa coupling via resonant s-channel Higgs production at FCC-ee*,
   Eur. Phys. J. Plus **137**, 201 (2022), arXiv:2107.02686.

2. I. Moult, L. Necib and J. Thaler,
   *New angles on energy correlation functions*,
   JHEP **12** (2016) 153, arXiv:1609.07483.

3. H. Qu and L. Gouskos,
   *ParticleNet: Jet Tagging via Particle Clouds*,
   Phys. Rev. D **101**, 056019 (2020), arXiv:1902.08570.

---

# Author

**Hugo Leigh-Watts**

MSc Nuclear and Particle Physics
University of Edinburgh

