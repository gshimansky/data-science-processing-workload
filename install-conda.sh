#!/bin/bash -e

CONDA_DIR=${HOME}/miniconda
CONDA_ENV_NAME=data-science-processing-workload

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p "${CONDA_DIR}" -f -u
rm -f Miniconda3-latest-Linux-x86_64.sh

source ${CONDA_DIR}/bin/activate
conda create -n ${CONDA_ENV_NAME} --yes -c conda-forge modin-all ray-dashboard scikit-learn-intelex xgboost

echo Execute the following command in your terminal to be able to run launcher script:
echo source ${CONDA_DIR}/bin/activate ${CONDA_ENV_NAME}
echo To permanently enable conda package manage in your bash shell execute:
echo ${CONDA_DIR}/bin/conda init bash

