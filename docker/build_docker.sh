#!/bin/bash

set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

NAME="ube-mbrl-camera-ready"
VERSION=1

DOCKER_BUILDKIT=1 docker build \
  --pull \
  --file Dockerfile \
	--tag "$NAME:latest" \
	--tag "$NAME:$VERSION" \
	"$@" \
	"$SCRIPT_DIR/../"
