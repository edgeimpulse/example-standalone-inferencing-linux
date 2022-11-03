# import tensorflow as tf
import numpy as np
import json
import sys
import os, cv2
import time

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(dir_path, 'resources/libraries'))
# import ei_tensorflow.inference as ei_inference

import tensorflow as tf
import tflite_runtime.interpreter as tflite_runtime

interpreter = tflite_runtime.Interpreter(model_path=os.path.join(dir_path, 'beer-i8.lite'))
interpreter.allocate_tensors()

def get_features_from_img(interpreter, img):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    print('img shape', img.shape)

    count, width, height, channels = input_shape

    print('count', count, 'width', width, 'height', height, 'channels', channels)

    if (channels == 3):
        return np.array([ x / 255 for x in list(img.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    elif (channels == 1):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img_grayscale = np.dot(img[...,:3], rgb_weights)
        return np.array([ x / 255 for x in list(img_grayscale.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    else:
        raise ValueError('Unknown depth for image')

def process_input(input_details, data):
    """Prepares an input for inference, quantizing if necessary.

    Args:
        input_details: The result of calling interpreter.get_input_details()
        data (numpy array): The raw input data

    Returns:
        A tensor object representing the input, quantized if necessary
    """
    if input_details[0]['dtype'] is np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        data = (data / scale) + zero_point
        data = np.around(data)
        data = data.astype(np.int8)

    return data
    # return tf.convert_to_tensor(data)

def invoke(interpreter, item: np.ndarray, specific_input_shape: 'list[int]'):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    item_as_tensor = process_input(input_details, item)
    if specific_input_shape:
        item_as_tensor = np.reshape(item_as_tensor, specific_input_shape)
    # Add batch dimension
    item_as_tensor = np.expand_dims(item_as_tensor, 0)
    interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output, output_details

def current_ms():
    return round(time.time() * 1000)

img = cv2.imread(os.path.join(dir_path, 'beer.jpg'))
input_data = get_features_from_img(interpreter, img)

print('img shape', img.shape)

start = current_ms()
res = invoke(interpreter, input_data, img.shape)
end = current_ms()
print('res', res, 'took', end - start, 'ms.')
