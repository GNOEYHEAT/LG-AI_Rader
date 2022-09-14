#!/bin/bash

# PATH Setting
script="`readlink -f "${BASH_SOURCE[0]}"`"
HOMEDIR="`dirname "$script"`"

source ${HOMEDIR}/pyenv/bin/activate

python ${HOMEDIR}/src/preprocess.py
python ${HOMEDIR}/src/inference.py
