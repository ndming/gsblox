# sqlite3 v3.24
add_subdirectory(sqlite3)

# nvblox v0.0.8
set(BUILD_PYTORCH_WRAPPER OFF)
set(BUILD_TESTING         OFF)
set(USE_SYSTEM_GFLAGS      ON)
set(USE_SYSTEM_EIGEN       ON)
set(USE_SYSTEM_BENCHMARK   ON)
set(USE_SYSTEM_GLOG        ON)
add_subdirectory(nvblox)

# gsplat v1.5.3
add_subdirectory(gsplat)

# pangolin v0.9.4
set(BUILD_TOOLS    OFF)
set(BUILD_EXAMPLES OFF)
add_subdirectory(pangolin)
