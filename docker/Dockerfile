FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG CUML_VERSION=v24.06.00

RUN apt-get update \
    && apt-get install -y \
    sudo \
    vim \
    less \
    git \
    wget \
    libssl-dev \
    autoconf \
    libtool \
    devscripts \
    debhelper \
    libblas-dev \
    liblapack-dev \
    zlib1g \
    cython3 \
    cuda-toolkit-12-2 \
    clang \
    ccache \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV USE_CCACHE=1
ENV CCACHE_DIR=/ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV PATH=/usr/lib/ccache:$PATH

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

# install ucx
RUN git clone --recursive https://github.com/openucx/ucx.git \
    && cd ucx \
    && ./autogen.sh \
    && ./contrib/configure-release \
    && make -j$(nproc) install \
    && cd .. \
    && rm -rf ucx

# install nccl for gtx 1080
RUN git clone https://github.com/NVIDIA/nccl.git \
    && cd nccl \
    && make -j$(nproc) pkg.debian.build NVCC_GENCODE="-gencode=arch=compute_61,code=sm_61" \
    && dpkg -i build/pkg/deb/* \
    && cd .. \
    && rm -rf nccl

# ref. https://github.com/rapidsai/cuml/issues/2528#issuecomment-656847070
RUN wget https://anaconda.org/nvidia/libcumlprims/24.06.00/download/linux-64/libcumlprims-24.06.00-cuda12_240605_gfa5d8ef_0.tar.bz2 \
    && mkdir -p /tmp/libcumlprims/ \
    && tar -xf libcumlprims-24.06.00-cuda12_240605_gfa5d8ef_0.tar.bz2 -C /tmp/libcumlprims/ \
    && cp -R /tmp/libcumlprims/include/* /usr/local/include/ \
    && cp -R /tmp/libcumlprims/lib/* /usr/local/lib/ \
    && rm -rf /tmp/libcumlprims/ libcumlprims-24.06.00-cuda12_240605_gfa5d8ef_0.tar.bz2 \
    && git clone https://github.com/rapidsai/cuml.git -b ${CUML_VERSION} \
    && cd cuml/cpp \
    && mkdir build && cd build \
    && cmake .. \
    -DDISABLE_DEPRECATION_WARNINGS=ON \
    -DUSE_CCACHE=ON \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    && make install \
    && cd ../../.. \
    && rm -r cuml 

RUN git clone https://github.com/gabime/spdlog.git -b v1.11.0 \
    && cd spdlog \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd ../.. \
    && rm -r spdlog
