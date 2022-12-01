import argparse
import paddle
from ..utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('model_dir',                   str,    None,         "inference model directory.")
add_arg('save_dir',                    str,    None,         "directory to save compressed model.")
add_arg('model_filename',              str,    None,         "inference model filename.")
# yapf: enable

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    paddle.enable_static()
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(paddle.CPUPlace())

    [inference_program, feed_target_names,
     fetch_targets] = paddle.static.load_inference_model(
         dirname=args.model_dir,
         executor=exe,
         model_filename=args.model_filename,
         params_filename=None)

    feed_vars = [
        inference_program.global_block().var(name) for name in feed_target_names
    ]
    paddle.static.save_inference_model(
        args.save_dir,
        executor=exe,
        model_filename='model.pdmodel',
        params_filename='model.pdiparams',
        feed_vars=feed_vars,
        fetch_vars=fetch_targets)
