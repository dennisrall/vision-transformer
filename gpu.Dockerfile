FROM tensorflow/tensorflow:2.4.2-gpu as base
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

FROM base AS pyenv
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        python-openssl \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        && apt-get purge -y --auto-remove \
        && rm -rf /var/lib/apt/lists/*
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN echo 'eval "$(pyenv init -)"' >> /root/.bashrc
RUN pyenv install 3.7.9 && pyenv global 3.7.9 && pip install poetry==1.1.6
ENV PATH=/root/.pyenv/versions/3.7.9/bin:$PATH
# Some TF tools expect a "python" binary
RUN rm /usr/local/bin/python && ln -s $(which python3) /usr/local/bin/python

FROM pyenv AS python-environment
COPY *.toml *.lock ./
RUN poetry config virtualenvs.create false && poetry install
WORKDIR /src

FROM python-environment AS dvc-repro
COPY . .
RUN chmod +x dvc-repro.sh
ENTRYPOINT ["/src/dvc-repro.sh"]
