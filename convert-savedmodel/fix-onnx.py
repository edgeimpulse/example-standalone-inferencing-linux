import onnx
import argparse

parser = argparse.ArgumentParser(description='Fix ONNX file')
parser.add_argument('--in-file', type=str, required=True)
parser.add_argument('--out-file', type=str, required=True)

args = parser.parse_args()

model = onnx.load(args.in_file)
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1
onnx.save(model, args.out_file)
