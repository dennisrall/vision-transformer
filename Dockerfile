FROM python:3.7.9-slim AS base
RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/* \
&& pip install poetry==1.1.6

FROM base AS python-environment
COPY *.toml *.lock ./
RUN poetry config virtualenvs.create false && poetry install
WORKDIR /src

FROM python-environment AS flake8
COPY . .
RUN flake8 src

FROM python-environment AS pytest
COPY . .
RUN TEST=True pytest src

FROM python-environment AS dvc-repro
COPY . .
RUN chmod +x dvc-repro.sh
ENTRYPOINT ["/src/dvc-repro.sh"]
