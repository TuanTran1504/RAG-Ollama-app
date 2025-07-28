FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y \
    wget git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set working directory and copy files
WORKDIR /app
COPY environment.yml .

# Accept ToS and install environment
RUN conda config --set always_yes yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults conda && \
    conda env create -f environment.yml


# Install PyTorch with CUDA manually
RUN conda run -n rag-env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy rest of the app
COPY . .

# Make startup script executable if needed
RUN chmod +x start.sh

# Set entrypoint
CMD ["./start.sh"]

