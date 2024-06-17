FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
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
    && apt-get clean

COPY install.sh /usr/local/bin/install.sh
RUN chmod +x /usr/local/bin/install.sh

WORKDIR /CNN_CPP

RUN /usr/local/bin/install.sh

RUN pip3 install invoke

VOLUME ["/CNN_CPP"]

ENTRYPOINT ["/bin/bash"]
