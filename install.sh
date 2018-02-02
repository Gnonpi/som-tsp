#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}

# Install matplotlib
#apt-get install python3-matplotlib

# Install llvmlite
mkdir -p dependencies
cd dependencies
git clone https://github.com/numba/llvmlite
cd llvmlite
python setup.py install

cd ${DIR}

# Install dependencies via Pypi
pip install -r requirements.txt
