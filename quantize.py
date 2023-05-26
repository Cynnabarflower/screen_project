import tensorflow as tf
import numpy as np

from oxford_dataset import OxfordPetDataset, IMAGE_WIDTH, IMAGE_HEIGHT


def representative_dataset():
    # data = tf.data.Dataset.from_tensor_slices(list(map(lambda a: a['image'], train_dataset))).batch(1).take(100)
    data = list(map(lambda a: a['image'], test_dataset))
    for input_value in data:
        w = input_value.shape[1]
        h = input_value.shape[2]
        r = np.zeros((1, w, h, 3))
        # print(r.shape)
        # print(input_value.shape)
        for x in range(w):
            for y in range(h):
                r[0][x][y] = np.array([input_value[0][x][y], input_value[1][x][y], input_value[2][x][y]],
                                      dtype='float32')
        yield [np.array(r, dtype='float32')]


# init train, val, test sets
root = "input_data"
# train_dataset = OxfordPetDataset(root, "train")
# valid_dataset = OxfordPetDataset(root, "valid")
test_dataset = OxfordPetDataset(root, "test")

# IMAGE_WIDTH = 480
# IMAGE_HEIGHT = 480
ONNX_MODEL_PATH = 'model.onnx'
WORKING_DIR = '.'
MODEL_NAME = f'model_{IMAGE_WIDTH}x{IMAGE_HEIGHT}'
openvino2tensorflow_out_dir = f'{WORKING_DIR}/openvino2tensorflow'

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.lite.TFLiteConverter.from_saved_model('openvino2tensorflow')
tflite_float16_model_path = f'{WORKING_DIR}/{MODEL_NAME}.float16.tflite'
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.representative_dataset = representative_dataset
print(f'Generating {tflite_float16_model_path} ...')
tflite_quant_model = converter.convert()
with open(tflite_float16_model_path, 'wb') as f:
    f.write(tflite_quant_model)
print('float16 done')


converter = tf.lite.TFLiteConverter.from_saved_model('openvino2tensorflow')
tflite_float16_model_path = f'{WORKING_DIR}/{MODEL_NAME}.int8.tflite'
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # We only want to use int8 kernels
converter.inference_input_type = tf.int8  # Can also be tf.int8
converter.inference_output_type = tf.int8  # Can also be tf.int8
converter.representative_dataset = representative_dataset
print(f'Generating {tflite_float16_model_path} ...')
tflite_quant_model = converter.convert()
with open(tflite_float16_model_path, 'wb') as f:
    f.write(tflite_quant_model)
print('int8 done')
