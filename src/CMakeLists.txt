add_library(MSFO MSFO.cu)

target_include_directories(MSFO
                           PRIVATE
                           ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/third_party/cuCollections/include)

target_link_libraries(MSFO 
                      PRIVATE 
                      CUDA::cudart
                      CUDA::cuda_driver)

set_property(TARGET MSFO PROPERTY CUDA_ARCHITECTURES native)