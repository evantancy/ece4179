#!/usr/bin/env bash
set -e

source /home/evan/miniconda3/etc/profile.d/conda.sh

conda create --name ece4179 python=3.8 -y && \
conda activate ece4179

conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install -y numpy matplotlib pandas scipy

pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
