# TAE-MoE: Resolving Multiscale Convergence Conflicts in Tropical Cyclone Forecasting

This repository contains the code for our KDD submission on tropical cyclone (TC) multi-task forecasting with **TAE-MoE (Task-Aware Equilibrium Mixture-of-Experts)**. The model jointly predicts **future TC track** and **intensity-related variables** while addressing the convergence conflict between fast-converging kinematic learning and slow-converging thermodynamic learning.

According to the paper, TAE-MoE is designed to reallocate expert capacity based on task convergence state and improves both trajectory and intensity forecasting across six ocean basins (EP, NA, NI, SI, SP, WP).

> **Important note on filenames**  
> If you downloaded files that contain suffixes such as `-2`, `-3`, etc., please remove these suffixes. They are only download artifacts and are **not** part of the actual filenames used by the codebase.

---

## 1. Overview

Tropical cyclone forecasting is a coupled multi-task problem: trajectory prediction is mainly governed by large-scale steering flow and typically converges with shorter temporal context, while intensity prediction depends on slower thermodynamic accumulation and benefits from longer context. Our paper formulates this mismatch as a **convergence conflict** and introduces a task-aware MoE framework to mitigate it.

In the released implementation, the model combines:

- a trajectory encoder-decoder,
- a 3D U-Net branch for TC-centered gridded fields,
- an environmental feature encoder,
- and a routing / generator-selection mechanism for multi-expert prediction.

---

## 2. Repository structure

The code imports modules using the `TCNM` package name, so the recommended layout is:

```text
repo_root/
├── README.md
├── scripts/
│   ├── train_github_4to4.py
│   ├── test_4to4.py
│   └── test_4to4_all.py
└── TCNM/
    ├── __init__.py
    ├── models_prior_unet4to4.py
    ├── env_net_transformer_gphsplit.py
    ├── Unet3D_merge_tiny4to4.py
    ├── losses.py
    ├── utils.py
    └── data/
        ├── __init__.py
        ├── loader_training4to4.py
        └── trajectoriesWithMe_unet_training4to4.py
```

### Filename normalization

Please make sure the actual filenames are the clean names below:

- `models_prior_unet4to4.py`
- `env_net_transformer_gphsplit.py`
- `Unet3D_merge_tiny4to4.py`
- `losses.py`
- `utils.py`
- `loader_training4to4.py`
- `trajectoriesWithMe_unet_training4to4.py`
- `train_github_4to4.py`
- `test_4to4.py`
- `test_4to4_all.py`

If your local file is currently named something like `utils-3.py`, rename it to `utils.py` before release.

---

## 3. Requirements

Recommended environment:

- Python 3.9+
- PyTorch
- NumPy
- OpenCV (`opencv-python`)

A minimal installation example is:

```bash
pip install torch numpy opencv-python
```

You will also need a CUDA-enabled PyTorch installation if you want to train on GPU.

---

## 4. Data

To make the repository lightweight and easier to release, we do **not** redistribute the raw datasets here. Please download the original data from the official sources and preprocess them into the layout expected by the code.

### 4.1 Best-track data

Official IBTrACS download page:

- https://www.ncei.noaa.gov/products/international-best-track-archive

### 4.2 Atmospheric reanalysis data

For the gridded atmospheric fields used in this project, please download ERA5 from the official Copernicus Climate Data Store.

Recommended ERA5 page for geopotential-height-style pressure-level fields:

- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download

If you use 500 hPa geopotential-height fields (as expected by the current code), please prepare them as TC-centered `.npy` grids.

### 4.3 What the released code expects

The released code expects **preprocessed** local files with the following components:

1. **Sequence text files** for train/val/test splits.
2. **TC-centered GPH `.npy` files**.
3. **Environmental feature `.npy` files**.

The expected local structure is:

```text
DATA_ROOT/
├── bst_divi10_train_val_test_inlcude15_2023_new/
│   ├── EP/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── NA/
│   ├── NI/
│   ├── SI/
│   ├── SP/
│   └── WP/
├── all_ocean_gph500_2023/
│   └── {AREA}/{YEAR}/{STORM_NAME}/{YYYYMMDDHH}.npy
└── all_area_correct_location_includeunder15_2023/
    └── {AREA}/{YEAR}/{STORM_NAME}/{YYYYMMDDHH}.npy
```

### 4.4 Expected split-file format

Each sample file under

```text
bst_divi10_train_val_test_inlcude15_2023_new/{AREA}/{train|val|test}/
```

is read as a tab-separated text file. The loader assumes:

- all columns except the last two are numeric,
- the **last two columns are metadata strings**,
- and these metadata are later used to recover the storm name and timestamp for loading the corresponding `.npy` files.

### 4.5 Expected environmental feature keys

The environment encoder expects the following keys in the environmental `.npy` dictionary:

- `wind`
- `intensity_class`
- `move_velocity`
- `month`
- `location_long`
- `location_lat`
- `history_direction12`
- `history_direction24`
- `history_inte_change24`

An optional `location` field may exist but is ignored by the current loader.

---

## 5. Paths you must replace before release or reproduction

This is the most important section for users.

### A. Training dataset root

In `train_github_4to4.py`, the training and validation splits are loaded from `--dataset_root`.

Recommended usage:

```bash
python train_github_4to4.py --obs_len=X --pred_len=Y --dataset_root /path/to/bst_divi10_train_val_test_inlcude15_2023_new
```

If you prefer not to pass it every time, you may edit the default path in the script.

### B. GPH root and environmental-feature root

In `TCNM/data/trajectoriesWithMe_unet_training4to4.py`, the dataset class contains **hard-coded local paths** inside `get_img()`:

```python
root = r'/your/local/data/root'
env_root = os.path.join(root, 'all_area_correct_location_includeunder15_2023')
modal_root = os.path.join(root, 'all_ocean_gph500_2023')
```

You **must** change `root` to your own local data directory.

This is the key replacement that users often miss.

### C. Evaluation dataset root and checkpoint path

In `test_4to4.py`, replace:

- `DEFAULT_MODEL_PATH`
- `DEFAULT_DATASET_ROOT`

or simply pass them from the command line.

Example:

```bash
python test_4to4.py \
  --obs_len=X \
  --pred_len=Y \
  --model_path /path/to/checkpoint.pt \
  --dataset_root /path/to/bst_divi10_train_val_test_inlcude15_2023_new
```

### D. Multi-checkpoint evaluation script

In `test_4to4_all.py`, replace:

- `DEFAULT_MODEL_DIR`
- `DEFAULT_DATASET_ROOT`

or pass them from the command line.

### E. Optional pretrained initialization

If you plan to use any pretrained initialization branch that depends on:

```text
pretrain_model/MMSTN_finetune.pt
```

please make sure the file exists locally or disable that branch in your own workflow.

---

## 6. Training

The main training command is:

```bash
python train_github_4to4.py --obs_len=X --pred_len=Y
```

A more complete example is:

```bash
python train_github_4to4.py \
  --obs_len 4 \
  --pred_len 4 \
  --dataset_root /path/to/bst_divi10_train_val_test_inlcude15_2023_new \
  --gpu_num 0 \
  --areas EP NA NI SI SP WP
```

Notes:

- `obs_len` and `pred_len` control the observation and prediction horizons.
- The current code supports variable observation length (for example `4/6/8`) and variable prediction length.
- If `--output_dir` is not specified, the script automatically builds an output directory that includes `obs_len` and `pred_len`.

---

## 7. Evaluation

### Single checkpoint

```bash
python test_4to4.py \
  --obs_len 4 \
  --pred_len 4 \
  --model_path /path/to/checkpoint_with_model_05000.pt \
  --dataset_root /path/to/bst_divi10_train_val_test_inlcude15_2023_new
```

### Sweep multiple checkpoints

```bash
python test_4to4_all.py \
  --obs_len 4 \
  --pred_len 4 \
  --model_dir /path/to/checkpoint_directory \
  --dataset_root /path/to/bst_divi10_train_val_test_inlcude15_2023_new
```

### Important evaluation note

The evaluation script checks that the checkpoint was trained with the same `obs_len` and `pred_len` that you pass at test time. Please make sure they match.

---

## 8. Practical release notes

Before making the repository public, we recommend the following cleanup:

1. Remove machine-specific absolute paths.
2. Confirm that all duplicate download suffixes (`-2`, `-3`, etc.) are removed from filenames.
3. Ensure the package layout matches the import statements under `TCNM/`.
4. Add a `LICENSE` file.
5. Add preprocessing scripts if you want full end-to-end reproducibility from raw IBTrACS and ERA5.

---

## 9. Citation

If you use this code in your research, please cite our work.

```bibtex
@misc{taemoe_code,
  title={TAE-MoE: Resolving Multiscale Convergence Conflicts in Scientific Multi-Task Learning via Task-Aware Equilibrium},
  author={Zhaoran Feng and collaborators},
  year={2026},
  howpublished={GitHub repository}
}
```

---

## 10. Contact

For questions regarding the code or data preparation, please contact the authors listed in the paper.
