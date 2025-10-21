import numpy as np, os, time, sys, urllib.request
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import Image

use_npu = True if len(sys.argv) >= 2 and sys.argv[1] == '--use-npu' else False

# Use HTP backend of libQnnTFLiteDelegate.so (NPU) when --use-npu is passed in
experimental_delegates = []
if use_npu:
    experimental_delegates = [load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})]

# Load TFLite model and allocate tensors
interpreter = Interpreter(
    model_path='tflite_learn_161053_5.tflite',
    experimental_delegates=experimental_delegates
)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load, preprocess and quantize image
def load_image(path, input_shape):
    # Expected input shape: [1, height, width, channels]
    _, height, width, channels = input_shape

    # Load image
    img = Image.open(path).convert("RGB").resize((width, height))
    img_np = np.array(img, dtype=np.float32)
    # !! Normalize... this model is 0..1 scaled (no further normalization); but that depends on your model !!
    img_np = img_np / 255
    # Add batch dim
    img_np = np.expand_dims(img_np, axis=0)

    scale, zero_point = input_details[0]['quantization']  # (scale, zero_point); scale==0.0 -> unquantized

    # Quantize input if needed
    if input_details[0]['dtype'] == np.float32:
        return img_np
    elif input_details[0]['dtype'] == np.uint8:
        # q = round(x/scale + zp)
        q = np.round(img_np / scale + zero_point)
        return np.clip(q, 0, 255).astype(np.uint8)
    elif input_details[0]['dtype'] == np.int8:
        # Commonly zero_point ≈ 0 (symmetric), but use provided zp anyway
        q = np.round(img_np / scale + zero_point)
        return np.clip(q, -128, 127).astype(np.int8)
    else:
        raise Exception('Unexpected dtype: ' + str(input_details[0]['dtype']))

input_shape = input_details[0]['shape']
input_data = load_image('jan.jpg.1t3khg7h.b0879da4335a.jpg.2m79la0p.283171b614fc.jpg.2qpoohnp.f749eb2b2dfe.jpg', input_shape)

# Set tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# Get prediction
q_output = interpreter.get_tensor(output_details[0]['index'])
scale, zero_point = output_details[0]['quantization']
f_output = (q_output.astype(np.float32) - zero_point) * scale

print('f_output', f_output)
