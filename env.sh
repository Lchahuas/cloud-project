#!/bin/bash

export VERTEX_LOCATION=us-central1
export VERTEX_PROJECT_ID=durable-ring-405419
export SOLUTION_NAME=heart-prediction
export CONTAINER_IMAGE_REGISTRY=${VERTEX_LOCATION}-docker.pkg.dev/${VERTEX_PROJECT_ID}/vertex-images
export VERTEX_PIPELINE_ROOT=gs://${BUCKET_NAME}/PIPELINES/${SOLUTION_NAME}/pl-root
export VERTEX_SA_EMAIL=248873594880-compute@developer.gserviceaccount.com
export BUCKET_NAME=durable-ring-405419_cloudbuild
