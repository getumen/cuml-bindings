FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i -r 's@http://(jp\.)?archive\.ubuntu\.com/ubuntu/?@http://ftp.jaist.ac.jp/pub/Linux/ubuntu/@g' /etc/apt/sources.list

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
    cython \
    cuda-toolkit-11-2 \
    libthrust-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.3/cmake-3.27.3.tar.gz \
    && tar -zxf cmake-3.27.3.tar.gz \
    && cd cmake-3.27.3 \
    && ./bootstrap \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf cmake-3.27.3 cmake-3.27.3.tar.gz

# install ucx
RUN git clone --recursive  https://github.com/openucx/ucx.git \
    && cd ucx \
    && ./autogen.sh \
    && ./contrib/configure-release \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf ucx

# install nccl for gtx 1080
RUN git clone https://github.com/NVIDIA/nccl.git \
    && cd nccl \
    && make -j$(nproc) pkg.debian.build NVCC_GENCODE="-gencode=arch=compute_61,code=sm_61" \
    && dpkg -i build/pkg/deb/libnccl2_2.18.3-1+cuda11.2_amd64.deb build/pkg/deb/libnccl-dev_2.18.3-1+cuda11.2_amd64.deb \
    && cd .. \
    && rm -rf nccl

RUN git clone https://github.com/dmlc/treelite.git -b 2.0.0 \
    && cd treelite \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd ../.. \
    && rm -r treelite

RUN git clone --recursive https://github.com/NVIDIA/thrust.git -b 2.1.0 \
    && cd thrust \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd ../.. \
    && rm -r thrust

RUN git clone https://github.com/rapidsai/rmm.git -b v21.08.00 \
    && cd rmm \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd ../.. \
    && rm -r rmm

RUN git clone https://github.com/rapidsai/raft.git -b v21.08.00 \
    && cd raft/cpp \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install -j$(nproc) \
    && cd ../../.. \
    && rm -r raft

# ref. https://github.com/rapidsai/cuml/issues/2528#issuecomment-656847070
RUN wget https://anaconda.org/nvidia/libcumlprims/21.08.00/download/linux-64/libcumlprims-21.08.00-cuda11.2_g9c188ef_0.tar.bz2 \
    && mkdir -p /tmp/libcumlprims/ \
    && tar -xf libcumlprims-21.08.00-cuda11.2_g9c188ef_0.tar.bz2 -C /tmp/libcumlprims/ \
    && mv /tmp/libcumlprims/include/* /usr/local/include/ \
    && mv /tmp/libcumlprims/lib/* /usr/local/lib/ \
    && rm -rf /tmp/libcumlprims/ libcumlprims-21.08.00-cuda11.2_g9c188ef_0.tar.bz2 \
    && git clone https://github.com/rapidsai/cuml.git -b v21.08.00 \
    && cd cuml/cpp \
    && mkdir build \
    && cd build \
    && cmake .. -DBUILD_CUML_TESTS=OFF -DBUILD_CUML_MG_TESTS=OFF -DBUILD_PRIMS_TESTS=OFF \
    && make install -j $(nproc) \
    && cd ../../.. \
    && rm -r cuml 
