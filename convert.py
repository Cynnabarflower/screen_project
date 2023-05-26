IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def convert():
    import onnxsim
    import onnx

    simplified_onnx_model, success = onnxsim.simplify('model.onnx')
    assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
    simplified_onnx_model_path = f'model.simplified.onnx'

    print(f'Generating {simplified_onnx_model_path} ...')
    onnx.save(simplified_onnx_model, simplified_onnx_model_path)
    print('done')

    import sys
    import os

    # Import the model optimizer tool from the openvino_dev package
    from openvino.tools.mo import main as mo_main
    import onnx
    from onnx_tf.backend import prepare
    # from mltk.utils.shell_cmd import run_shell_cmd

    # Load the ONNX model
    onnx_model = onnx.load('model.onnx')
    tf_rep = prepare(onnx_model)

    # Get the input tensor shape
    input_tensor = tf_rep.signatures[tf_rep.inputs[0]]
    input_shape = input_tensor.shape
    input_shape_str = '[' + ','.join([str(x) for x in input_shape]) + ']'

    openvino_out_dir = f'openvino'
    os.makedirs(openvino_out_dir, exist_ok=True)

    print(f'Generating openvino at: {openvino_out_dir}')
    cmd = [
        sys.executable, os.path.relpath(mo_main.__file__, '.'),
        '--input_model', '/Users/dmitry/PycharmProjects/screen_project/model.simplified.onnx',
        '--input_shape', f'{input_shape_str}',
        '--output_dir', openvino_out_dir,
        '--data_type', 'FP32'

    ]

    # command = r"python3 ./venv/lib/python3.9/site-packages/openvino/tools/mo/main.py --input_model " \
    #           r"model.simplified.onnx --input_shape '[1, 3, 256, 256]' --output_dir openvino --data_type FP32"
    command = 'python3 '+' '.join(cmd[1:])
    print(command)
    stream = os.popen(command)
    output = stream.read()
    print(output)

    # retcode, retmsg = run_shell_cmd(cmd, outfile=sys.stdout)
    # assert retcode == 0, 'Failed to do conversion'

    import os
    # from mltk.utils.shell_cmd import run_shell_cmd

    openvino2tensorflow_out_dir = f'openvino2tensorflow'
    openvino_xml_name = os.path.basename(simplified_onnx_model_path)[:-len('.onnx')] + '.xml'

    if os.name == 'nt':
        openvino2tensorflow_exe_cmd = [sys.executable,
                                       os.path.join(os.path.dirname(sys.executable), 'openvino2tensorflow')]
    else:
        openvino2tensorflow_exe_cmd = ['openvino2tensorflow']

    print(f'Generating openvino2tensorflow model at: {openvino2tensorflow_out_dir} ...')
    cmd = openvino2tensorflow_exe_cmd + [
        '--model_path', f'{openvino_out_dir}/{openvino_xml_name}',
        '--model_output_path', openvino2tensorflow_out_dir,
        '--output_saved_model',
        '--output_no_quant_float32_tflite'
    ]

    # openvino2tensorflow - -model_path
    # openvino / model.simplified.xml - -model_output_path
    # openvino2tensorflow - -output_saved_model - -output_no_quant_float32_tflite

    # print(' '.join(cmd))
    command = ' '.join(cmd)
    print(command)
    stream = os.popen(command)
    output = stream.read()
    print(output)
    print('done')

    # import onnx

    # onnx_model = onnx.load('model.onnx')
    # # Convert with onnx-tf:

    # from onnx_tf.backend import prepare

    # tf_rep = prepare(onnx_model)
    # # Export TF model:

    # tf_rep.export_graph('model.pb')

    # converter = tf.lite.TFLiteConverter.from_saved_model('model.pb')
    # converter.target_spec.supported_ops = [
    #   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    #   # tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    # ]
    # tflite_model = converter.convert()

    # # Save the model
    # with open('tflite_model_path.tflite', 'wb') as f:
    #     f.write(tflite_model)

convert()
