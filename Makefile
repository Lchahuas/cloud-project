-include env.sh
export
    
pre-commit: ## Runs the pre-commit checks over entire repo
	cd pipelines && \
	poetry run pre-commit run --all-files

install: ## Set up local environment for Python development on pipelines
	@cd ml-model && \
	poetry install --with dev && \
	cd utilities && \
	poetry install --with dev

compile: ## Compile the pipeline to pipeline.yaml. Must specify pipeline=<training|prediction|features>
	@cd pipelines/src && \
	poetry run kfp dsl compile --py pipelines/${pipeline}.py --output pipelines/${pipeline}.yaml --function pipeline ;

compile ?= true
cache ?= true
wait ?= false
run: ## Run pipeline in sandbox environment. Must specify pipeline=<training|prediction>. Optionally specify wait=<true|false> (default = false). Set compile=false to skip recompiling the pipeline.
	@if [ $(compile) = "true" ]; then \
		$(MAKE) compile; \
	elif [ $(compile) != "false" ]; then \
		echo "ValueError: compile must be either true or false" ; \
		exit ; \
	fi && \
	cd pipelines/src && \
	ENABLE_PIPELINE_CACHING=$$cache poetry run python -m pipelines.utils.trigger_pipeline \
	--template_path=pipelines/${pipeline}.yaml --display_name=${pipeline} --wait=${wait};

training: ## Run training pipeline. Supports same options as run.
	@$(MAKE) run pipeline=training

prediction:	## Run prediction pipeline. Supports same options as run.
	@$(MAKE) run pipeline=prediction

features:	## Run features pipeline. Supports same options as run.
	@$(MAKE) run pipeline=features