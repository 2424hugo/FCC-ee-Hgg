# FCC-ee H → gg Signal vs Background Analysis

## Overview

This repository contains an analysis framework for studying Higgs boson decays to two gluons (H → gg) at the FCC-ee.

The aim of the project is to identify observables capable of discriminating gluon-initiated Higgs decays from the dominant quark-antiquark background, providing the foundation for future machine learning classification.

The analysis is performed using Delphes simulated events and Awkward Arrays, with ROOT files cached into parquet format for efficient processing.

Current work includes:

- ROOT → parquet caching
- Event-level selection and optimisation
- Jet-level analysis
- Two-jet event reconstruction
- Jet substructure observables
- Energy Correlation Functions (ECFs)
- Simple multivariate classification studies
- ROC and AUC performance evaluation

---

# Dataset

The analysis uses FCC-ee Delphes samples stored on CERN EOS.

## Signal

```
e⁺e⁻ → H → gg
```

## Background

```
e⁺e⁻ → q\bar{q}
```

Only the branches required for the analysis are cached locally.

---

# Project Structure

```text
FCC-ee-Hgg/
├── cache/
├── outputs/
│   └── plots/
│       ├── 2_jet_selection/
│       ├── cut_data/
│       ├── energy_func/
│       └── ...
├── scripts/
│   ├── cutting_data/
│   ├── 2_jet_selection/
│   ├── classifier/
│   ├── hl_observables/
│   ├── make_cache.py
│   ├── plots_jet.py
│   ├── plots_recon.py
│   ├── plots_tracks.py
│   └── plots_others.py
├── config.py
├── setup.sh
└── README.md
```

---

# Environment Setup

Load the CERN LCG environment

```bash
source setup.sh
```

This loads

- Python
- ROOT
- uproot
- awkward
- NumPy
- matplotlib
- scikit-learn

from

```bash
/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt
```

---

# Creating Local Caches

Convert the Delphes ROOT files into parquet caches

```bash
python scripts/make_cache.py
```

The caches are stored in

```
cache/
```

and are ignored by Git.

---

# Analysis Workflow

## 1. Basic Object Distributions

Generate standard jet and event distributions

```bash
python scripts/plots_jet.py
python scripts/plots_recon.py
python scripts/plots_tracks.py
python scripts/plots_others.py
```

---

## 2. Event Selection

Apply and optimise event invariant mass cuts

```bash
python scripts/cutting_data/finding_cut.py
python scripts/cutting_data/mass_cutts.py
```

---

## 3. Two-Jet Selection

Select the two highest-energy jets and study their discriminating power

```bash
python scripts/2_jet_selection/two_highest_energy.py
python scripts/2_jet_selection/regression_2jet_masses.py
```

This includes

- leading/subleading jet studies
- ROC curves
- logistic regression
- decision trees
- 2D jet-mass distributions

---

## 4. Energy Correlation Functions

Implemented observables include

- 2-point ECF
- 3-point ECF
- 1e3
- C2
- D2

with multiple β values.

Scripts are located in

```text
scripts/hl_observables/
```

---

# Physics Observables

## Event-Level

- Invariant mass
- Missing energy
- Missing momentum

---

## Jet-Level

- Energy
- Mass
- Momentum
- Charge
- Constituent multiplicity
- Leading jet variables
- Subleading jet variables

---

## Jet Substructure

- Energy fractions
- Pairwise angular distances
- Two-point Energy Correlation Function
- Three-point Energy Correlation Function
- 1e3
- C2
- D2

---

# Classification Studies

Current classification studies include

- ROC curves
- Area Under Curve (AUC)
- Cut optimisation
- Logistic Regression
- Decision Trees

These studies provide baseline performance before implementing more sophisticated machine learning methods.

---

# Outputs

Generated plots are stored in

```text
outputs/plots/
```

including

- cut optimisation
- jet studies
- energy correlation functions
- ROC curves
- classifier comparisons

---

# Future Work

Planned developments include

- ParticleNet implementation
- Transformer-based jet classification
- Additional jet substructure observables
- Event-level machine learning
- Hyperparameter optimisation
- Feature importance studies
- Significance optimisation
- Full Higgs event classification

---

# Author

**Hugo Leigh-Watts**

MSc Nuclear and Particle Physics

University of Edinburgh
