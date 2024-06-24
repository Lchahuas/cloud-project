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

# Usa la imagen base de Python 3.10.13
FROM python:3.10.13 AS builder

# Configuración de entorno para pip
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

COPY requirements.txt requirements.txt

# Instala las dependencias
RUN pip install -r requirements.txt

# Etapa de entrenamiento
FROM builder AS training

WORKDIR /training

# Copia los archivos de código fuente específicos del modelo
COPY src/training/ training/

# Etapa de predicción
FROM builder AS prediction

# Instala las dependencias necesarias para la predicción
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copia el archivo main.py específico del modelo de predicción
COPY src/prediction/main.py prediction/main.py

# Comando por defecto para ejecutar la aplicación
CMD exec uvicorn prediction.main:app --host "0.0.0.0" --port 8000