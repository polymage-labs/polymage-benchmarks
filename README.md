# polymage-benchmarks

This repository contains certain benchmarks and workloads to share with
PolyMage-external contacts and collaborators to help reproduce performance. The
benchmarks are assorted but are mostly from the field of deep learning and
benchmarked from Python-based AI programming frameworks like PyTorch and
TensorFlow.

To benchmark GPU kernel execution times, as an example:

```shell
$ nsys profile -s none --cpuctxsw=none --trace=cuda -o gpu_ python tensorflow/conv2d.py
$ nsys stats -q --report gpukernsum --format table gpu_.nsys-rep --timeunit=milliseconds
```

When benchmarking JIT compiled execution, a typical strategy is to run a certain
number times and then that many times + desired number of iterations for
benchmarking, and subtract the former from the latter. Either an average of the
result over the runs or the median can be taken. Eg., run 10 times and then 110
times and take the difference and average it over 100 runs.
