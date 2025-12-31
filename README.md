# GsBlox: Gaussian splatting + NVIDIA NvBlox

```bash
mkdir build
cd build
cmake .. -DTorch_DIR=<path/to/libtorch>/share/cmake/Torch -DCMAKE_CUDA_ARCHITECTURES=120
make -j10
```
