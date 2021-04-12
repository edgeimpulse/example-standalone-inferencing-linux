#!/bin/bash
set -e

export DOCKER_BUILDKIT=1

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

echo 'Creating ONNX model...'
python3 -u -m tf2onnx.convert --saved-model saved_model --output model.onnx
python3 fix-onnx.py --in-file model.onnx --out-file ../model-1x1-batch.onnx
echo 'Creating ONNX model OK'
echo ''
