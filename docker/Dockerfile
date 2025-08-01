FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basics
RUN apt update && apt install -y \
    wget git ffmpeg libsm6 libxext6 \
    build-essential  # ← includes gcc, g++, make

ENV CONDA_DIR /opt/conda

# RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
#     /bin/bash ./Anaconda3-2021.11-Linux-x86_64.sh -b -p /opt/conda && \
#     rm ./Anaconda3-2021.11-Linux-x86_64.sh 

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh && \
    bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py39_4.12.0-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
# --- Create Python environment ---
RUN conda install -y -c conda-forge python=3.9 && conda clean -afy

# --- Install fixed NumPy version ---
RUN pip install numpy

# --- Install PyTorch 2.7.0 + CUDA 12.8 ---
RUN pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# --- Install compatible torch-scatter ---
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# --- Additional packages ---
RUN pip install hydra-core numba spconv-cu120 \
    opencv-python nuscenes-devkit protobuf==3.20.* \
    scipy matplotlib

RUN pip install numpy==1.24.4

# Optional: set working directory
# WORKDIR /workspace