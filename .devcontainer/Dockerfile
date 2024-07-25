FROM rapidsai/base:24.06-cuda12.2-py3.11

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update \
    && apt-get install -y \
    vim \
    less \
    git \
    wget \
    libssl-dev \
    build-essential \
    cuda-toolkit-12-2 \
    clang \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CPATH=/opt/conda/include:/opt/conda/include/rapids:/usr/local/include:/usr/local/cuda/include
ENV LIBRARY_PATH=$LIBRARY_PATH:/opt/conda/lib:/opt/conda/lib/rapids:/usr/local/lib:/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib:/opt/conda/lib/rapids:/usr/local/lib:/usr/local/cuda/lib64
    
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.3/cmake-3.27.3.tar.gz \
    && tar -zxf cmake-3.27.3.tar.gz \
    && cd cmake-3.27.3 \
    && ./bootstrap \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -r cmake-3.27.3 cmake-3.27.3.tar.gz

RUN git clone https://github.com/dmlc/treelite.git -b 4.1.2 \
    && cd treelite \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd ../.. \
    && rm -r treelite

RUN git clone https://github.com/gabime/spdlog.git -b v1.11.0 \
    && cd spdlog \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd ../.. \
    && rm -r spdlog