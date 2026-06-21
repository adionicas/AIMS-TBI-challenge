#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Default container name
container_tag="example-algorithm-detection"

# Allow an override as the first argument
if [ "$#" -eq 1 ]; then
    container_tag="$1"
fi

# Confirm the image exists
build_timestamp=$( docker inspect --format='{{ .Created }}' "$container_tag")
if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $container_tag"
    exit 1
fi

formatted_build_info=$(date +"%Y%m%d_%H%M%S")
output_filename="${SCRIPT_DIR}/${container_tag}_${formatted_build_info}.tar.gz"

# Save and gzip the image
docker save "$container_tag" | gzip -c > "$output_filename"

echo "Container saved as ${output_filename}"
