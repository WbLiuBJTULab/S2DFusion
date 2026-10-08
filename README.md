## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL> S2DFusion
cd S2DFusion
export S2D_ROOT=$PWD
```

### 2. Create the Environment

```bash
conda create -n S2Dfusion python=3.8 -y
conda activate S2Dfusion
```

### 3. Install PyTorch and Dependencies

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    pytorch-cuda=11.8 -c pytorch -c nvidia

pip install spconv-cu118
pip install -r requirements.txt
```

Install the recommended versions for VoD evaluation:

```bash
pip install --no-cache-dir \
    numpy==1.23.5 \
    llvmlite==0.39.1 \
    numba==0.56.4
```

### 4. Build OpenPCDet Extensions

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"

cd "$S2D_ROOT"
python setup.py develop
```

### 5. Install Mamba

```bash
cd "$S2D_ROOT/mamba"
pip install -e .
```

If the prebuilt wheel is unavailable:

```bash
MAMBA_FORCE_BUILD=TRUE pip install -e .
```

---

## VoD Dataset Preparation

Download the **View-of-Delft (VoD)** dataset following the official instructions.

The expected directory structure is:

```text
rlfusion_5f/
├── ImageSets/
│   ├── train.txt
│   └── val.txt
├── training/
│   ├── calib/
│   ├── image_2/
│   ├── label_2/
│   ├── lidar/
│   ├── radar/
│   ├── radar_5f/
│   └── radar_calib/
└── testing/
```

Input point formats:

- **LiDAR:** `x, y, z, intensity`
- **4D Radar:** `x, y, z, rcs, v_r, v_r_comp, time`

Generate the dataset information files and ground-truth database:

```bash
cd "$S2D_ROOT"

python -m pcdet.datasets.vod.vod_dataset \
    create_vod_infos tools/cfgs/dataset_configs/Vod_fusion.yaml
```

The following files will be generated:

```text
vod_infos_train.pkl
vod_infos_val.pkl
vod_infos_trainval.pkl
vod_dbinfos_train.pkl
gt_database/
```

---

## Training

All training commands should be executed from the `tools/` directory.

### Single-GPU Training

```bash
cd "$S2D_ROOT/tools"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg_file cfgs/VoD_models/S2DFusion.yaml \
    --batch_size 4 \
    --workers 8 \
    --extra_tag S2Dfusion_vod
```

### Multi-GPU Training

Example using four GPUs:

```bash
cd "$S2D_ROOT/tools"

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/dist_train.sh 4 \
    --cfg_file cfgs/VoD_models/S2DFusion.yaml \
    --batch_size 4 \
    --workers 8 \
    --extra_tag S2Dfusion_vod
```

---

## Evaluation

Evaluate a trained checkpoint with:

```bash
cd "$S2D_ROOT/tools"

CUDA_VISIBLE_DEVICES=0 python test.py \
    --cfg_file cfgs/VoD_models/S2DFusion.yaml \
    --batch_size 4 \
    --workers 4 \
    --extra_tag S2Dfusion_vod \
    --ckpt /path/to/checkpoint.pth \
    --eval_tag test
```

---

## Acknowledgements

This repository is built upon the following open-source projects and datasets:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [Mamba](https://github.com/state-spaces/mamba)
- [spconv](https://github.com/traveller59/spconv)
- [View-of-Delft Dataset](https://github.com/tudelft-iv/view-of-delft-dataset)

We sincerely thank the authors for making their code and datasets publicly available.

Please follow the corresponding licenses and citation requirements when using this repository.
