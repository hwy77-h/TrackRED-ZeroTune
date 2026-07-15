FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore \
    ZEROTUNE_DEVICE=cuda \
    ZEROTUNE_SAM2_ROOT=/opt/algorithm/sam2 \
    ZEROTUNE_SAM2_CHECKPOINT=/opt/ml/model/sam2.1_hiera_small.pt \
    ZEROTUNE_WORK_DIR=/tmp/zerotune_work

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

RUN mkdir -p /opt/algorithm && \
    git clone https://github.com/facebookresearch/sam2.git /opt/algorithm/sam2

WORKDIR /opt/algorithm/sam2
RUN python -m pip install -e .

RUN python -m pip install \
      SimpleITK \
      pillow \
      scipy \
      scikit-image \
      numpy==1.26.3 \
      hydra-core==1.3.2 \
      iopath==0.1.10

RUN groupadd -r user && useradd -m -r -g user user

WORKDIR /opt/algorithm/zerotune
COPY --chown=user:user . /opt/algorithm/zerotune

RUN mkdir -p /output/images/mri-linac-series-targets /tmp/zerotune_work && \
    chown -R user:user /opt/algorithm/zerotune /output /tmp/zerotune_work

USER user

ENTRYPOINT ["python", "inference_GC.py"]
