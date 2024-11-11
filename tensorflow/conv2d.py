# Benchmark a typically deep learning convolution with various parameters. The
# benchmark is a proxy for the performance of libraries like CuDNN on NVIDIA
# GPUs.
import argparse
import sys
import time
import numpy as np
import tensorflow as tf

### Conv2d run via TensorFlow's standard executor in graph mode.
@tf.function
def conv2d(inputs, filters, stride, padding, data_format, dilations):
    return tf.nn.conv2d(inputs, filters, stride, padding, data_format, dilations)

### Conv2d run via TensorFlow XLA.
@tf.function(jit_compile=True)
def conv2d_xla(inputs, filters, stride, padding, data_format, dilations):
    return conv2d(inputs, filters, stride, padding, data_format, dilations)

### Create and initialize input and filters.
def init(input_shape, filter_shape):
    inputs = np.random.rand(
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ).astype(np.float16)
    filters = np.random.rand(
        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]
    ).astype(np.float16)
    return [inputs, filters]

if len(tf.config.list_physical_devices('GPU')) == 0:
    print("No CUDA visible GPUs available; consider installing TensorFlow "
          "with GPU support")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-input-shape",
    nargs="+",
    type=int,
    default=[1, 224, 224, 3],
    help="Input shape in N, H, W, C format",
)
parser.add_argument(
    "-filter-shape",
    nargs="+",
    type=int,
    default=[5, 5, 3, 3],
    help="Filter shape",
)
parser.add_argument(
    "-strides",
    nargs="+",
    type=int,
    default=[1, 1],
    help="strides in each of the spatial dimension",
)
parser.add_argument(
    "-dilations",
    nargs="+",
    type=int,
    default=[1, 1],
    help="dilation in each of the spatial dimension",
)
parser.add_argument(
    "-padding",
    type=str,
    default="VALID",
    help="padding type, can be either SAME or VALID",
)
parser.add_argument(
    "-data-format",
    type=str,
    default="NHWC",
    help="data format, can be either NHWC or NCHW",
)
parser.add_argument(
    "-xla",
    action="store_true",
    default=False,
    help="Use XLA (off by default)",
)
parser.add_argument(
    "-num-iters",
    type=int,
    default=100,
    help="Number of timing iterations",
)

args = vars(parser.parse_args())
input_shape = args.get("input_shape")
filter_shape = args.get("filter_shape")
stride = args.get("strides")
dilations = args.get("dilations")
padding = args.get("padding")
data_format = args.get("data_format")
xla = args.get("xla")
num_iters = args.get("num_iters")

inputs, filters = init(input_shape, filter_shape)

if xla:
    # Run a few times and measure execution time.
    start_time = time.time()
    print(f"Running {num_iters} iterations via TF/XLA...")
    for _ in range(num_iters):
        conv2d_xla(inputs, filters, stride, padding, data_format, dilations)
    end_time = time.time()

    # Print execution time in milliseconds.
    xla_exec_time = ((end_time - start_time) * 1000)/num_iters
    print(f"TF/XLA average execution time per iteration in ms: {xla_exec_time}")
else:
    # Run a few times and measure execution time.
    start_time = time.time()
    print(f"Running {num_iters} runs via TF/standard...")
    for _ in range(num_iters):
        conv2d(inputs, filters, stride, padding, data_format, dilations)
    end_time = time.time()

    tf_std_exec_time = ((end_time - start_time) * 1000)/num_iters
    print("TF/standard average execution time per iteration in ms: "
          f"{tf_std_exec_time}")
