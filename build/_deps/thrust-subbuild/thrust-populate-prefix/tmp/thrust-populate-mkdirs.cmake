# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-src"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-build"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-subbuild/thrust-populate-prefix"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-subbuild/thrust-populate-prefix/tmp"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-subbuild/thrust-populate-prefix/src/thrust-populate-stamp"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-subbuild/thrust-populate-prefix/src"
  "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-subbuild/thrust-populate-prefix/src/thrust-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-subbuild/thrust-populate-prefix/src/thrust-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/fangzhouai/MultiShardFusedOperation/build/_deps/thrust-subbuild/thrust-populate-prefix/src/thrust-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
