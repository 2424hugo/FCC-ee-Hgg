# FCC-ee `H → gg` versus `e⁺e⁻ → q\bar{q}`

An event-level jet-substructure study of direct, resonant Higgs production at the FCC-ee. The analysis distinguishes the `H → gg` final state from continuum hadronic `e⁺e⁻ → q\bar{q}` events using simulated Delphes/IDEA samples, energy-correlation functions (ECFs), and a boosted decision tree (BDT).

The wider motivation is sensitivity to the electron Yukawa coupling through `e⁺e⁻ → H` at `√s ≈ m_H`. This channel is experimentally difficult because the direct production rate is very small and hadronic backgrounds are much larger. A generator-level study identified `H → gg` as one of the more promising final states for this programme [d’Enterria, Poldaru & Wojcik (2022)](https://arxiv.org/abs/2107.02686).

## Analysis summary

- **Signal:** `e⁺e⁻ → H → gg`
- **Background:** `e⁺e⁻ → q\bar{q}`
- **Detector simulation:** FCC-ee Winter 2023 Delphes events with the IDEA detector card
- **Selection:** jet-based event invariant mass `m_event > 120 GeV`, followed by the two highest-energy jets in each retained event
- **Classifier:** 22-variable `HistGradientBoostingClassifier`, trained with class-balanced weights

The project is an analysis-development study. The BDT weights are deliberately balanced for classification and are **not** cross-section or luminosity weights; a final physics sensitivity calculation must apply the relevant rates, luminosity, selection efficiencies, and systematic uncertainties.

## Current validated BDT result

The committed reference run used 100,000 signal and 100,000 background events for both training and validation, with a held-out test split left untouched.

| Quantity | Validation result |
| --- | ---: |
| ROC AUC | 0.9358 |
| Average precision | 0.9283 |
| Youden threshold | 0.4910 |
| Signal efficiency at this threshold | 0.8818 |
| Background efficiency at this threshold | 0.1593 |
| Background rejection | 0.8407 |
| Balanced accuracy | 0.8612 |

The corresponding model, metrics, ROC curve and permutation-importance plot are written to `outputs/ml/bdt_22_variables/`.

The strongest variables in the reference permutation test were the leading and subleading `C₂(β=0.2)` observables, followed by leading-jet mass, constituent multiplicity and `D₂(β=0.2)`.

## Repository layout

```text
FCC-ee-Hgg/
├── setup.sh
├── scripts/
│   ├── data_processing/
│   │   ├── add_hl_observables.py
│   │   └── enrich_analysis_shards.py
│   ├── data_checks/
│   │   └── validate_enriched_shards.py
│   └── ML/
│       └── train_bdt_22_variables.py
├── cache/                         # generated data; not version-controlled
└── outputs/
    ├── ml/bdt_22_variables/
    └── plots/
```

Older exploratory scripts for cache construction, cut studies, jet distributions and ECF scans are retained under `scripts/`.

## Environment

On CERN LXPLUS, initialise the project environment from the repository root:

```bash
source setup.sh
```

This loads the CERN LCG 108 environment and sets `PYTHONPATH` to the repository root. The analysis relies on Python, `uproot`, `awkward`, NumPy, PyArrow, Matplotlib, scikit-learn and joblib.

## Data pipeline

The input samples are EDM4hep/Delphes ROOT files on CERN EOS. The cache builder reads only the branches required for the analysis in batches, reconstructs the jet-based event four-vector, and stores selected events as Parquet shards.

Each retained event has:

1. `m_event > 120 GeV`;
2. at least two reconstructed jets;
3. two jets ordered by energy (leading, then subleading);
4. constituent four-vectors resolved through the `Jet#2` relation; and
5. a fixed signal (`1`) or background (`0`) label.

The analysis dataset is organised as:

```text
cache/analysis_dataset/
├── signal/{train,validation,test}/*.parquet
└── background/{train,validation,test}/*.parquet
```

Parquet files are read in batches rather than loaded eagerly. This is necessary for the full cache, which exceeds typical interactive memory limits.

### Add high-level observables

The enrichment stage calculates the selected observables in small event chunks and atomically replaces a shard only after it has been reloaded and validated:

```bash
python -m scripts.data_processing.enrich_analysis_shards \
  --all \
  --dataset-root cache/analysis_dataset/signal/train \
  --chunk-size 500
```

Repeat for each sample and split. To process one shard, use `--file path/to/shard.parquet` instead of `--all`.

Validate all enriched shards and the train/validation/test source-file separation with:

```bash
python -m scripts.data_checks.validate_enriched_shards
```

`C₂` and `D₂` are undefined for one-constituent jets because `e₂ = e₃ = 0`; these entries are stored as `NaN` and are handled by the BDT.

## Observables

The BDT uses 22 scalar inputs: two event-level quantities plus ten values for each selected jet.

| Category | Variables |
| --- | --- |
| Event | `event_invariant_mass`, `n_jets_original` |
| Per selected jet | energy, mass, constituent multiplicity, `p_T`, `p`, polar angle `θ` |
| Jet substructure | `e₂(β=0.2)`, `e₃(β=0.2)`, `C₂=e₃/e₂²`, `D₂=e₃/e₂³` |

For constituents with energy fractions `z_i`, the ECFs use the energy-normalised angular distance `θ_ij`:

`e₂^(β) = Σ_{i<j} z_i z_j θ_ij^β`

`e₃^(β) = Σ_{i<j<k} z_i z_j z_k (θ_ij θ_ik θ_jk)^β`

Earlier single-observable ECF3 scans gave AUCs of 0.8328 (`β=0.1`), 0.8255 (`0.2`), 0.7975 (`0.5`), 0.7587 (`1.0`) and 0.7124 (`2.0`). The current enriched dataset therefore uses `β=0.2`, which provides strong discrimination in combination with the other observables.

## Train the BDT

Run the reference configuration from the repository root:

```bash
python -m scripts.ML.train_bdt_22_variables \
  --dataset-root cache/analysis_dataset \
  --max-events-per-class 100000 \
  --importance-events 20000
```

The script:

- reads only training and validation shards, never the test split;
- derives leading/subleading scalar columns from the two-jet arrays;
- uses `HistGradientBoostingClassifier` with a maximum of 300 iterations, 31 leaves per tree, minimum leaf size 100, L2 regularisation of 1.0 and ROC-AUC early stopping;
- applies class-balanced training weights; and
- writes `bdt_model.joblib`, `metrics.json`, `validation_roc.png` and `permutation_importance.png`.

Use `--max-events-per-class 0` only when sufficient memory and runtime are available to load all events in each split.

## Reproducibility and data handling

- Generated caches and large analysis outputs are excluded from Git.
- Dataset splits are defined at the source-file level to prevent event leakage across train, validation and test samples.
- The enrichment and validation steps explicitly check event counts, two-jet shapes, finite physical quantities and the expected `C₂`/`D₂` `NaN` pattern.
- The supplied model result is a validation result, not a final unbiased performance estimate. Evaluate the reserved test split once model selection and threshold optimisation are finalised.

## References

1. D. d’Enterria, A. Poldaru and G. Wojcik, *Measuring the electron Yukawa coupling via resonant s-channel Higgs production at FCC-ee*, Eur. Phys. J. Plus **137**, 201 (2022), [arXiv:2107.02686](https://arxiv.org/abs/2107.02686).
2. A. J. Larkoski, I. Moult and D. Neill, *Power Counting to Better Jet Observables*, JHEP **12** (2014) 009, [arXiv:1409.6298](https://arxiv.org/abs/1409.6298).  

## Author

Hugo Leigh-Watts  
MSc Nuclear and Particle Physics, University of Edinburgh
