# Semantics-Preserving Temporal Adversarial Graph Contrastive Learning for Event Prediction

This repository contains the implementation of **Sead**, proposed in the paper
**"Semantics-Preserving Temporal Adversarial Graph Contrastive Learning for Event Prediction"**.
The paper has been accepted by the **ECML-PKDD 2026 Research Track**.

## Overview

Sead addresses multi-event prediction on temporal event graphs. Given a historical
window of events, the model predicts the distribution of event types that may occur
at a future timestamp.

The key idea is to learn robust event representations without destroying semantic
associations between events. Sead contrasts the clean temporal event graph with
three auxiliary views:

- **Semantics-preserving view**: preserves event associations using contextual
  semantic similarity.
- **Adversarial view**: introduces worst-case structural and feature perturbations
  to improve robustness.
- **Temporal perturbation view**: perturbs temporal event features to handle time
  noise in real-world data.

These views are encoded by a semantics-aware event encoder, which combines adaptive
graph aggregation, CompGCN-style message passing, and temporal encoding. The final
event-type representations are optimized with graph contrastive loss and event
prediction loss.

## Repository Structure

```text
Sead/
|-- README.md
|-- __init__.py
`-- src/
    |-- train.py                 # training, validation, and testing
    |-- build_baseline.py        # argument extension and model construction
    |-- models.py                # Sead model
    |-- aggregators.py           # event graph aggregation modules
    |-- propagations.py          # graph propagation modules
    |-- Generator.py             # adversarial learning component
    |-- Discriminator.py         # adversarial learning component
    |-- data.py                  # dataset wrappers
    |-- event_data_processing.py # event data preprocessing
    |-- utils.py                 # data loading, metrics, and utilities
    |-- modules_f.py             # attention and basic neural modules
    `-- modules/                 # memory/cache/TGN-related modules
```

## Usage

```bash
python src/train.py --model Sead --dataset <DATASET_NAME> --dp <DATA_ROOT>/
```

The expected dataset directory is:

```text
<DATA_ROOT>/<DATASET_NAME>/
|-- stat.txt
|-- train.txt
|-- valid.txt
|-- test.txt
`-- dg_dict.txt
```

The event files use the quadruple format:

```text
head relation tail time
```

## Notes

The experiments in the paper are conducted on GDELT event datasets for Egypt,
Iran, and Israel. The current repository contains source code only; datasets,
preprocessed graph dictionaries, and dependency files are not included.

Main dependencies include PyTorch, DGL, NumPy, SciPy, scikit-learn, pandas,
tqdm, torch-scatter, and matplotlib.
