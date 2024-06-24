# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM python:3.9.16 AS builder

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

ARG POETRY_VERSION=1.6.1

COPY utilities/ utilities/
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN pip install poetry==${POETRY_VERSION}
RUN poetry config virtualenvs.create false

FROM builder AS training

ARG MODEL_NAME

WORKDIR /training

RUN poetry install
COPY ${MODEL_NAME}/training/src ./src/

FROM builder AS serving

ARG MODEL_NAME

RUN poetry install --with serving
COPY ${MODEL_NAME}/serving/main.py serving/main.py

CMD exec uvicorn serving.main:app --host "0.0.0.0" --port "$AIP_HTTP_PORT"