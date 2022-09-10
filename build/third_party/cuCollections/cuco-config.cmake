#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

#[=======================================================================[

Provide targets for cuCollections.

cuCollections (cuco) is an open-source, header-only library of GPU-accelerated,
concurrent data structures.

Similar to how Thrust and CUB provide STL-like, GPU accelerated algorithms and
primitives, cuCollections provides STL-like concurrent data structures.
cuCollections is not a one-to-one, drop-in replacement for STL data structures
like std::unordered_map. Instead, it provides functionally similar data
structures tailored for efficient use with GPUs.



Result Variables
^^^^^^^^^^^^^^^^

This module will set the following variables::

  CUCO_FOUND
  CUCO_VERSION
  CUCO_VERSION_MAJOR
  CUCO_VERSION_MINOR

#]=======================================================================]


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../../usr/local" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

cmake_minimum_required(VERSION 3.18)

set(rapids_global_languages )
foreach(lang IN LISTS rapids_global_languages)
  include("${CMAKE_CURRENT_LIST_DIR}/cuco-${lang}-language.cmake")
endforeach()
unset(rapids_global_languages)

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cuco-dependencies.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cuco-dependencies.cmake")
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cuco-targets.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cuco-targets.cmake")
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cuco-config-version.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cuco-config-version.cmake")
endif()

# Set our version variables
set(CUCO_VERSION_MAJOR 0)
set(CUCO_VERSION_MINOR 0)
set(CUCO_VERSION_PATCH 1)
set(CUCO_VERSION 0.0.1)


set(rapids_global_targets cuco)
set(rapids_namespaced_global_targets cuco)
if(rapids_namespaced_global_targets)
  list(TRANSFORM rapids_namespaced_global_targets PREPEND cuco:: )
endif()

foreach(target IN LISTS rapids_namespaced_global_targets)
  if(TARGET ${target})
    get_target_property(_is_imported ${target} IMPORTED)
    get_target_property(_already_global ${target} IMPORTED_GLOBAL)
    if(_is_imported AND NOT _already_global)
      set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()

# For backwards compat
if("rapids_config_build" STREQUAL "rapids_config_build")
  foreach(target IN LISTS rapids_global_targets)
    if(TARGET ${target})
      get_target_property(_is_imported ${target} IMPORTED)
      get_target_property(_already_global ${target} IMPORTED_GLOBAL)
      if(_is_imported AND NOT _already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()
      if(NOT TARGET cuco::${target})
        add_library(cuco::${target} ALIAS ${target})
      endif()
    endif()
  endforeach()
endif()

unset(rapids_global_targets)
unset(rapids_namespaced_global_targets)

check_required_components(cuco)

set(${CMAKE_FIND_PACKAGE_NAME}_CONFIG "${CMAKE_CURRENT_LIST_FILE}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME} CONFIG_MODE)

if(NOT TARGET cuco::Thrust)
thrust_create_target(cuco::Thrust FROM_OPTIONS)
endif()

