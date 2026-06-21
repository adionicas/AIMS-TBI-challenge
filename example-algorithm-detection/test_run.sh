#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="example-algorithm-detection"
DOCKER_NOOP_VOLUME="${DOCKER_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= Cleaning up any earlier output"
if [ -d "$OUTPUT_DIR" ]; then
  rm -rf "${OUTPUT_DIR}"/*
  chmod -f o+rwx "$OUTPUT_DIR"
else
  mkdir -m=o+rwx "$OUTPUT_DIR"
fi

echo "=+= (Re)build the container"
docker build "$SCRIPT_DIR" \
  --platform=linux/amd64 \
  --tag $DOCKER_TAG 2>&1

echo "=+= Doing a forward pass"
# '--network none' means no internet connection is available
# '--volume <NAME>:/tmp' is used because /tmp cannot store permanent files on Grand Challenge
docker volume create "$DOCKER_NOOP_VOLUME"
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --volume "$INPUT_DIR":/input \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    $DOCKER_TAG
docker volume rm "$DOCKER_NOOP_VOLUME"

# Restore host ownership of the output files
docker run --rm \
    --quiet \
    --env HOST_UID=`id -u` \
    --env HOST_GID=`id -g` \
    --volume "$OUTPUT_DIR":/output \
    alpine:latest \
    /bin/sh -c 'chown -R ${HOST_UID}:${HOST_GID} /output'

echo "=+= Wrote results to ${OUTPUT_DIR}"
echo "=+= Save this image for uploading via save.sh \"${DOCKER_TAG}\""
