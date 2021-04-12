#!/bin/bash
set -e

export DOCKER_BUILDKIT=1

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

if [[ "$OSTYPE" == "darwin"* ]]; then
    SEDCMD="sed -i '' -e"
    LC_CTYPE=C
    LANG=C
else
    SEDCMD="sed -i -e"
fi

cd $SCRIPTPATH

echo 'Creating ONNX model...'
python3 -u -m tf2onnx.convert --saved-model saved_model --output model.onnx
python3 fix-onnx.py --in-file model.onnx --out-file trained.onnx
xxd -i trained.onnx > ../tflite-model/onnx-trained.h
$SEDCMD "s/unsigned char/const unsigned char/g" ../tflite-model/onnx-trained.h
echo '// Empty on purpose' > ../tflite-model/onnx-trained.cpp
rm -f "../tflite-model/onnx-trained.h''"

echo 'Creating ONNX model OK'
echo ''
