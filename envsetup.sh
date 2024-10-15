#!/bin/bash
# Install dependency for fairseq

# Name of the conda environment
ENVNAME=grad_cam

eval "$(conda shell.bash hook)"
conda activate ${ENVNAME}
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Install conda environment ${ENVNAME}"
    
    # conda env
    conda create -n ${ENVNAME} python=3.9 pip=23.3.2 --yes
    conda activate ${ENVNAME}
    echo "===========Install pytorch==========="
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

    # install numpy
    pip install numpy==1.26.4
    # install scipy
    pip install scipy==1.7.3

    # install pandas
    pip install pandas==1.3.5

    # install protobuf
    pip install protobuf==3.20.3

    # install tensorboard
    pip install tensorboard==2.6.0
    pip install tensorboardX==2.6

    # install librosa
    pip install librosa==0.10.0

    # install pydub
    pip install pydub==0.25.1

    # install pyyaml
    pip install pyyaml

    # install tqdm
    pip install tqdm

    # install asteroid-filterbanks
    pip install asteroid-filterbanks

    # install einops
    pip install einops

    # install speechbrain
    pip install speechbrain

    # install torchinfo
    pip install torchinfo

    # install fairseq
    pip install git+https://github.com/pytorch/fairseq@a54021305d6b3c4c5959ac9395135f63202db8f1

    # install grad_cam
    pip install grad-cam
else
    echo "Conda environment ${ENVNAME} has been installed"
fi
