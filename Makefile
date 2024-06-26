# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

-include env.sh
export

ALL_MODELS := $(shell cd model && find . -maxdepth 1 -type d ! -name "." ! -name "utilities" -exec basename {} \;)

help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
    
pre-commit: ## Runs the pre-commit checks over entire repo
	cd pipelines && \
	poetry run pre-commit run --all-files

env ?= dev

targets ?= training prediction
models ?= src
build: 
	@cd ml-model && \
	if [ "$$models" = "*" ]; then \
		models=$(ALL_MODELS); \
	fi; \
	for model in $$models; do \
		for target in $$targets ; do \
			echo "Building $$target image for $$model model" && \
			gcloud builds submit . \
			--region=${VERTEX_LOCATION} \
			--project=${VERTEX_PROJECT_ID} \
			--gcs-source-staging-dir=gs://${VERTEX_PROJECT_ID}_cloudbuild/source \
			--substitutions=_DOCKER_TARGET=$$target,_DESTINATION_IMAGE_URI=${CONTAINER_IMAGE_REGISTRY}/${SOLUTION_NAME}-$$model-$$target:${RESOURCE_SUFFIX},_MODEL_NAME=$$model ; \
		done \
	done

compile:
	@cd pipelines && \
	kfp dsl compile --py ${pipeline}.py --output ${pipeline}.yaml --function pipeline;

run: 
	cd pipelines && \
	python -m utils.trigger_pipeline --template_path=${pipeline}.yaml --display_name=${pipeline} --wait=false;

test: ## Run unit tests for a specific component group or for all component groups and the pipeline trigger code. Optionally specify GROUP=<component group e.g. vertex-components>
	@if [ -n "${GROUP}" ]; then \
		echo "Test components under components/${GROUP}" && \
		cd components/${GROUP} && \
		poetry run pytest ; \
	elif [ -n "${MODEL}" ]; then \
		echo "Test components under model/${MODEL}" && \
		cd model/${MODEL}/training && \
		poetry run pytest ; \
	else \
		echo "Testing pipeline scripts" && \
		cd pipelines && \
		poetry run python -m pytest tests/utils && \
		cd .. && \
		for i in components/*/ ; do \
			echo "Test components under $$i" && \
			cd "$$i" && \
			poetry run pytest && \
			cd ../.. ; \
		done ; \
		echo "Testing utilities library" && \
		cd model/utilities && \
		poetry run python -m pytest && \
		cd ../.. && \
		for i in model/*/ ; do \
			if [ "$$i" != "model/utilities/" ]; then \
				echo "Test $$i model training scripts" && \
				cd "$$i/training/" && \
				poetry run pytest && \
				cd ../.. ; \
			fi \
		done ; \
	fi