# FCC-ee H → γγ Signal vs Background Analysis

## Overview

This repository contains an analysis framework for studying Higgs boson decays to two photons at the FCC-ee.

The project compares signal and background samples produced with Delphes and investigates kinematic and event-level observables that may be useful for signal discrimination and future machine learning classification.

Current work focuses on:

* Loading Delphes ROOT files with `uproot`
* Caching selected branches as parquet files
* Comparing signal and background distributions
* Feature engineering for future classification studies

---

## Dataset

The analysis uses FCC-ee Delphes samples stored on CERN EOS.

### Signal

Higgs boson decay:

H → γγ

### Background

Standard Model background samples from FCC-ee event generation.

Only a subset of branches is currently cached for rapid analysis.

---

## Project Structure

```text
FCC-ee-Hgg/
├── cache/
├── outputs/
│   └── plots/
├── scripts/
│   ├── make_cache.py
│   ├── plots_jet.py
│   ├── plots_recon.py
│   ├── plots_tracks.py
│   └── plots_others.py
├── config.py
├── setup.sh
├── .gitignore
└── README.md
```

### cache/

Stores locally cached parquet files generated from ROOT samples.

These files are not tracked by Git.

### outputs/

Stores generated plots and analysis outputs.

### scripts/

Contains data processing and plotting scripts.

---

## Environment Setup

Load the CERN LCG environment:

```bash
source setup.sh
```

The setup script loads:

* Python
* ROOT
* uproot
* awkward
* matplotlib
* scientific Python libraries

from:

```bash
/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt
```

---

## Generating Caches

Create local parquet caches from ROOT files:

```bash
python scripts/make_cache.py
```

This converts selected ROOT branches into awkward-parquet datasets for faster analysis.

---

## Plotting

Generate jet distributions:

```bash
python scripts/plots_jet.py
```

Generate reconstructed particle distributions:

```bash
python scripts/plots_recon.py
```

Generate track distributions:

```bash
python scripts/plots_tracks.py
```

Generate missing-energy, photon and neutral-hadron distributions:

```bash
python scripts/plots_others.py
```

Plots are written to:

```text
outputs/plots/
```

---

## Current Features

### Jet Variables

* Energy
* Mass
* Momentum components
* Charge
* Constituent multiplicity

### Reconstructed Particles

* Energy
* Mass
* Charge
* Momentum components

### Tracks

* D0
* Z0
* φ
* ω
* tan λ

### Missing Energy and Neutral Objects

* Missing energy
* Missing momentum components
* Photon energy
* Neutral hadron energy

---

## Future Work

Planned developments include:

* Additional high-level physics observables
* Jet constituent studies
* Quark/gluon discrimination variables
* Event shape observables
* Higgs candidate reconstruction
* Machine learning classification
* Feature importance studies
* Signal significance optimisation

---

## Author

Hugo Leigh-Watts

MSc Nuclear and Particle Physics

University of Edinburgh

