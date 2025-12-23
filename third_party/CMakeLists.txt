# nvblox v0.0.8
set(BUILD_PYTORCH_WRAPPER OFF)
set(BUILD_TESTING         OFF)
set(USE_SYSTEM_GFLAGS      ON)
set(USE_SYSTEM_EIGEN       ON)
set(USE_SYSTEM_BENCHMARK   ON)
set(USE_SYSTEM_GLOG        ON)
add_subdirectory(nvblox)

# gsplat v1.5.3
set(CAFFE2_USE_CUDNN      ON)
set(CAFFE2_USE_CUFILE     ON)
set(CAFFE2_USE_CUSPARSELT ON)
set(USE_CUDSS             ON)
set(USE_SYSTEM_NVTX       ON)
add_subdirectory(gsplat)
