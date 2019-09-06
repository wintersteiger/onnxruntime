#!/bin/bash
set -e
while getopts p: parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
esac
done

PYTHON_VER=${PYTHON_VER:=3.5}
DEBIAN_FRONTEND=noninteractive

SYS_LONG_BIT=$(getconf LONG_BIT)

apt-get update && apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update && apt-get install -y --no-install-recommends \
        autotools-dev \
        automake \
        build-essential \
        git apt-transport-https apt-utils \
        ca-certificates \
        pkg-config \
        wget \
        zlib1g \
        zlib1g-dev \
        libssl-dev \
        curl libcurl4-openssl-dev \
        autoconf \
        sudo \
        gfortran \
        python3-dev \
        language-pack-en \
        libopenblas-dev \
        liblttng-ust0 \
        libcurl3 \
        libssl1.0.0 \
        libkrb5-3 \
        libicu55 \
        libtinfo-dev \
        libtool \
        aria2 \
        bzip2 \
        unzip \
        zip \
        rsync libunwind8 libpng16-dev libexpat1-dev \
        python3-setuptools python3-numpy python3-wheel python python3-pip python3-pytest \
        libprotobuf-dev libprotobuf9v5 protobuf-compiler \
        libedit-dev libxml2-dev python3-packaging

locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

echo "Installing dotnet-sdk"
if [ $SYS_LONG_BIT = "64" ]; then
  OS_VER=`lsb_release -r -s`
  mkdir -p /tmp/dotnet
  aria2c -q -d /tmp/dotnet https://packages.microsoft.com/config/ubuntu/${OS_VER}/packages-microsoft-prod.deb
  dpkg -i /tmp/dotnet/packages-microsoft-prod.deb
  apt-get update
  apt-get install -y dotnet-sdk-2.2
  rm -rf /tmp/dotnet
fi

if [ $PYTHON_VER != "3.5" ]; then
    apt-get install -y --no-install-recommends \
            python${PYTHON_VER} \
            python${PYTHON_VER}-dev
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VER} 1
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 2
    update-alternatives --set python3 /usr/bin/python${PYTHON_VER}
    #TODO: the old one(/usr/bin/pip3) should be uninstalled first. Because the one will be
    #put at /usr/local/. Then there will be two pips.
    /usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall pip==19.0.3
fi

/usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall numpy==1.15.0
/usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall requests==2.21.0
/usr/bin/python${PYTHON_VER} -m pip install --upgrade --force-reinstall sympy==1.1.1
rm -rf /var/lib/apt/lists/*

aria2c -q -d /tmp -o llvm.tar.xz http://releases.llvm.org/6.0.1/clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz
tar --strip 1 -Jxf /tmp/llvm.tar.xz -C /usr
