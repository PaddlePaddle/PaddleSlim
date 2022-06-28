import os
import sys
import cv2
import numpy as np
import platform
import argparse
import base64
import shutil
import paddle
from paddleslim.auto_compression.utils.postprocess import build_postprocess
from paddleslim.auto_compression.utils.preprocess import create_operators
from paddleslim.auto_compression.config_helpers import load_config


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/config.yaml',
        help='config file path')
    return parser


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in args.items():
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


def get_image_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']
    if os.path.isfile(img_file) and img_file.split('.')[-1] in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            if single_file.split('.')[-1] in img_end:
                imgs_lists.append(os.path.join(img_file, single_file))
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


class Predictor(object):
    def __init__(self, config):
        predict_args = config['Global']
        # HALF precission predict only work when using tensorrt
        if predict_args['use_fp16'] is True:
            assert predict_args.use_tensorrt is True
        self.args = predict_args
        if self.args.get("use_onnx", False):
            self.predictor, self.config = self.create_onnx_predictor(
                predict_args)
        else:
            self.predictor, self.config = self.create_paddle_predictor(
                predict_args)

        self.preprocess_ops = []
        self.postprocess = None
        if "PreProcess" in config:
            if "transform_ops" in config["PreProcess"]:
                self.preprocess_ops = create_operators(config["PreProcess"][
                    "transform_ops"])
        if "PostProcess" in config:
            self.postprocess = build_postprocess(config["PostProcess"])

        # for whole_chain project to test each repo of paddle
        self.benchmark = config["Global"].get("benchmark", False)
        if self.benchmark:
            import auto_log
            import os
            pid = os.getpid()
            size = config["PreProcess"]["transform_ops"][1]["CropImage"]["size"]
            if config["Global"].get("use_int8", False):
                precision = "int8"
            elif config["Global"].get("use_fp16", False):
                precision = "fp16"
            else:
                precision = "fp32"
            self.auto_logger = auto_log.AutoLogger(
                model_name=config["Global"].get("model_name", "cls"),
                model_precision=precision,
                batch_size=config["Global"].get("batch_size", 1),
                data_shape=[3, size, size],
                save_path=config["Global"].get("save_log_path",
                                               "./auto_log.log"),
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=2)

    def create_paddle_predictor(self, args):
        inference_model_dir = args['inference_model_dir']

        params_file = os.path.join(inference_model_dir, args['params_filename'])
        model_file = os.path.join(inference_model_dir, args['model_filename'])

        config = paddle.inference.Config(model_file, params_file)

        if args['use_gpu']:
            config.enable_use_gpu(args['gpu_mem'], 0)
        else:
            config.disable_gpu()
            if args['enable_mkldnn']:
                # there is no set_mkldnn_cache_capatity() on macOS
                if platform.system() != "Darwin":
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(args['cpu_num_threads'])

        if args['enable_profile']:
            config.enable_profile()
        config.disable_glog_info()
        config.switch_ir_optim(args['ir_optim'])  # default true
        if args['use_tensorrt']:
            precision = paddle.inference.Config.Precision.Float32
            if args.get("use_int8", False):
                precision = paddle.inference.Config.Precision.Int8
            elif args.get("use_fp16", False):
                precision = paddle.inference.Config.Precision.Half

            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=args['batch_size'],
                workspace_size=1 << 30,
                min_subgraph_size=30,
                use_calib_mode=False)

        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)

        return predictor, config

    def create_onnx_predictor(self, args):
        import onnxruntime as ort
        inference_model_dir = args['inference_model_dir']
        model_file = os.path.join(inference_model_dir, args['model_filename'])
        config = ort.SessionOptions()
        if args['use_gpu']:
            raise ValueError(
                "onnx inference now only supports cpu! please specify use_gpu false."
            )
        else:
            config.intra_op_num_threads = args['cpu_num_threads']
            if args['ir_optim']:
                config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        predictor = ort.InferenceSession(model_file, sess_options=config)
        return predictor, config

    def predict(self, images):
        use_onnx = self.args.get("use_onnx", False)
        if not use_onnx:
            input_names = self.predictor.get_input_names()
            input_tensor = self.predictor.get_input_handle(input_names[0])

            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
        else:
            input_names = self.predictor.get_inputs()[0].name
            output_names = self.predictor.get_outputs()[0].name

        if self.benchmark:
            self.auto_logger.times.start()
        if not isinstance(images, (list, )):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)
        if self.benchmark:
            self.auto_logger.times.stamp()

        if not use_onnx:
            input_tensor.copy_from_cpu(image)
            self.predictor.run()
            batch_output = output_tensor.copy_to_cpu()
        else:
            batch_output = self.predictor.run(
                output_names=[output_names], input_feed={input_names: image})[0]

        if self.benchmark:
            self.auto_logger.times.stamp()
        if self.postprocess is not None:
            batch_output = self.postprocess(batch_output)
        if self.benchmark:
            self.auto_logger.times.end(stamp=True)
        return batch_output


def main(config):
    predictor = Predictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])
    image_list = image_list * 1000
    batch_imgs = []
    batch_names = []
    cnt = 0
    for idx, img_path in enumerate(image_list):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(
                "Image file failed to read and has been skipped. The path: {}".
                format(img_path))
        else:
            img = img[:, :, ::-1]
            batch_imgs.append(img)
            img_name = os.path.basename(img_path)
            batch_names.append(img_name)
            cnt += 1

        if cnt % config["Global"]["batch_size"] == 0 or (idx + 1
                                                         ) == len(image_list):
            if len(batch_imgs) == 0:
                continue
            batch_results = predictor.predict(batch_imgs)
            for number, result_dict in enumerate(batch_results):
                if "PersonAttribute" in config[
                        "PostProcess"] or "VehicleAttribute" in config[
                            "PostProcess"]:
                    filename = batch_names[number]
                else:
                    filename = batch_names[number]
                    clas_ids = result_dict["class_ids"]
                    scores_str = "[{}]".format(", ".join("{:.2f}".format(
                        r) for r in result_dict["scores"]))
                    label_names = result_dict["label_names"]
            batch_imgs = []
            batch_names = []
    if predictor.benchmark:
        predictor.auto_logger.report()
    return


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    config = load_config(args.config)
    print_arguments(config['Global'])
    main(config)
