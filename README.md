Installation

1. Clone the repository

git clone <YOUR_REPOSITORY_URL> S2DFusion
cd S2DFusion
export S2D_ROOT=$PWD

2. Create the environment

conda create -n S2Dfusion python=3.8 -y
conda activate S2Dfusion

3. Install PyTorch and dependencies

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    pytorch-cuda=11.8 -c pytorch -c nvidia

pip install spconv-cu118
pip install -r requirements.txt

pip install --no-cache-dir \
    numpy==1.23.5 \
    llvmlite==0.39.1 \
    numba==0.56.4

4. Build OpenPCDet extensions

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"

cd "$S2D_ROOT"
python setup.py develop

5. Install Mamba

cd "$S2D_ROOT/mamba"
pip install -e .

If the prebuilt wheel is unavailable:

MAMBA_FORCE_BUILD=TRUE pip install -e .



VoD Dataset Preparation

Download the View-of-Delft dataset following the official instructions:

View-of-Delft Dataset

The expected data structure is:

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

The input point formats are:

LiDAR: x, y, z, intensity

Radar: x, y, z, rcs, v_r, v_r_comp, time


Then generate the dataset information files and ground-truth database:

cd "$S2D_ROOT"

python -m pcdet.datasets.vod.vod_dataset \
    create_vod_infos tools/cfgs/dataset_configs/Vod_fusion.yaml

The generated files should include:

vod_infos_train.pkl
vod_infos_val.pkl
vod_infos_trainval.pkl
vod_dbinfos_train.pkl
gt_database/



Training

All training commands should be executed from the tools/ directory.

Single GPU

cd "$S2D_ROOT/tools"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg_file cfgs/VoD_models/S2DFusion.yaml \
    --batch_size 4 \
    --workers 8 \
    --extra_tag S2Dfusion_vod

Multi-GPU

Example with four GPUs:

cd "$S2D_ROOT/tools"

export CUDA_VISIBLE_DEVICES=0,1,2,3

bash scripts/dist_train.sh 4 \
    --cfg_file cfgs/VoD_models/S2DFusion.yaml \
    --batch_size 4 \
    --workers 8 \
    --extra_tag S2Dfusion_vod

Evaluation

Evaluate a checkpoint using:

cd "$S2D_ROOT/tools"

CUDA_VISIBLE_DEVICES=0 python test.py \
    --cfg_file cfgs/VoD_models/S2DFusion.yaml \
    --batch_size 4 \
    --workers 4 \
    --extra_tag S2Dfusion_vod \
    --ckpt /path/to/checkpoint.pth \
    --eval_tag test


    Acknowledgements

This repository is built upon the following open-source projects:

S2DFusion

OpenPCDet

Mamba

spconv

View-of-Delft Dataset

Please follow the licenses and citation requirements of the original projects and datasets when using this repository.
    
