# PolyMage Benchmarks

This repository contains certain benchmarks and workloads to share with
PolyMage-external contacts and collaborators to help reproduce performance. The
benchmarks are expected to be assorted, but mostly from the field of deep
learning and AI. They are executed from Python-based AI programming frameworks
like PyTorch and TensorFlow.

To benchmark kernel execution times on an NVIDIA GPU, as an example:

```shell
$ nsys profile -s none --cpuctxsw=none --trace=cuda -o gpu_ python tensorflow/conv2d.py
$ nsys stats -q --report gpukernsum --format table gpu_.nsys-rep --timeunit=milliseconds
```

When benchmarking JIT compiled execution, a typical strategy is to run a certain
number of times, and then again, that many times + a desired number of
iterations for benchmarking, and subtract the former from the latter. Either an
average of that result over the runs or the median can be taken. Eg., run 10
times and then 110 times, take the difference, and average it over 100 runs.
