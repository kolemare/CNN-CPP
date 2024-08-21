FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    wget \
    unzip \
    build-essential \
    libeigen3-dev \
    libopencv-dev \
    pkg-config \
    libjpeg-dev \
    libtiff-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-pip \
    doxygen \
    texlive-full \
    && apt-get clean

RUN pip3 install --upgrade pip
RUN pip3 install requests matplotlib invoke tensorflow keras numpy scipy

WORKDIR /CNN_CPP

VOLUME ["/CNN_CPP"]

ENTRYPOINT ["/bin/bash"]
