include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(TEST TEST.cu)

target_include_directories(TEST
                           PUBLIC
                           ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/third_party/cuCollections/include)

target_link_libraries(TEST PUBLIC GTest::gtest_main)
target_link_libraries(TEST PUBLIC MSFO CUDA::cudart)

gtest_discover_tests(TEST)
