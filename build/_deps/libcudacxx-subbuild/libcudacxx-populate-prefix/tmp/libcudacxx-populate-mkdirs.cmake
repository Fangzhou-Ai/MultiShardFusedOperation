# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-src"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-build"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-subbuild/libcudacxx-populate-prefix"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-subbuild/libcudacxx-populate-prefix/tmp"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-subbuild/libcudacxx-populate-prefix/src/libcudacxx-populate-stamp"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-subbuild/libcudacxx-populate-prefix/src"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-subbuild/libcudacxx-populate-prefix/src/libcudacxx-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-subbuild/libcudacxx-populate-prefix/src/libcudacxx-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/fangzhouai/MultiShardFusedOperation/build/_deps/libcudacxx-subbuild/libcudacxx-populate-prefix/src/libcudacxx-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
