include("/home/fangzhouai/MultiShardFusedOperation/build/cmake/CPM_0.35.5.cmake")
CPMAddPackage("NAME;libcudacxx;VERSION;1.8.0;GIT_REPOSITORY;https://github.com/NVIDIA/libcudacxx.git;GIT_TAG;1.8.0;GIT_SHALLOW;ON;EXCLUDE_FROM_ALL;OFF")
set(libcudacxx_FOUND TRUE)