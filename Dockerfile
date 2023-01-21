FROM ubuntu:22.04

# Proxy settings
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

RUN  DEBIAN_FRONTEND=noninteractive \
    apt-get update --yes \
    && apt-get install wget --yes \
    && rm -rf /var/lib/apt/lists/*

ENV USER modin
ENV UID 1000
ENV HOME /home/${USER}

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid ${UID} \
    --home ${HOME} \
    ${USER}
USER ${USER}

# Conda settings
ENV CONDA_DIR=${HOME}/miniconda
ENV CONDA_ENV_NAME=modin
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u \
    && "${CONDA_DIR}/bin/conda" init bash \
    && rm -f /tmp/miniconda3.sh

RUN conda update -n base -c defaults conda -y && conda install -n base conda-libmamba-solver
RUN conda create -n ${CONDA_ENV_NAME} --yes --experimental-solver=libmamba -c conda-forge \
    modin-all \
    scikit-learn-intelex \
    xgboost
RUN conda clean --all --yes

# Activate ${CONDA_ENV_NAME} for interactive shells
RUN echo "source ${CONDA_DIR}/bin/activate ${CONDA_ENV_NAME}" >> "${HOME}/.bashrc"
# Activate ${CONDA_ENV_NAME} for non-interactive shells
# The following line comments out line that prevents ~/.bashrc execution in
# non-interactive mode.
RUN sed -e 's,\(^[[:space:]]\+[*]) return;;$\),# \1,' -i "${HOME}/.bashrc"
ENV BASH_ENV="${HOME}/.bashrc"

# Set up benchmark scripts
WORKDIR ${HOME}
COPY . ${HOME}

# Clean up proxy settings to publish on Docker Hub
ENV http_proxy=
ENV https_proxy=
ENV no_proxy=

# Set entrypoint with arguments expansion
ENTRYPOINT ["/home/modin/miniconda/envs/modin/bin/python", "launcher.py"]
