
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

cmake_minimum_required(VERSION 3.12)

project(hipblaslt_test)

set(COMPILE_FOR_ROCM 1)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)

file(GLOB INC *.h *.hpp ../common/*.h ../common/*.hpp)
file(GLOB SRC *.cpp *.cc)
file(GLOB GPU_SRC *.cu.cc)

set(USE_EXTERNAL_HIPBLASLT 1)
set(HIPBLASLT_DIR "/tf/Playground/hipBLASLt/build/release/hipblaslt-install")

if(NOT ${COMPILE_FOR_ROCM})
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_VERBOSE_BUILD ON)
    #set(CUDA_NVCC_FLAGS "--ptxas-options;--maxrregcount=128;")
    set(CMAKE_CUDA_COMPILER ${CMAKE_CXX_COMPILER})
   
    # NVCC line
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --ptxas-options=-v -arch=compute_70 -code=sm_70")
    # CLANG line
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -Xcuda-ptxas -v --cuda-gpu-arch=sm_70")
    set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE CUDA)
else()
    #list(APPEND CMAKE_PREFIX_PATH "/opt/rocm-5.6.0/lib/cmake/AMDDeviceLibs")
    #list(APPEND CMAKE_PREFIX_PATH "/opt/rocm-5.6.0/lib/cmake/amd_comgr")
    list(APPEND CMAKE_MODULE_PATH "/opt/rocm-5.6.0/lib/cmake/hip") 

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nobuiltininc")

    enable_language(HIP) 
    find_package(HIP REQUIRED)
    #find_package(HIPBLASLT REQUIRED PATHS ${HIPBLASLT_DIR}/hipblaslt/lib/cmake)

    set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(${SRC} PROPERTIES LANGUAGE HIP)
endif()

if(${USE_EXTERNAL_HIPBLASLT})
    include_directories(${HIPBLASLT_DIR})
    add_definitions(-DUSE_EXTERNAL_HIPBLASLT)
    link_directories(${HIPBLASLT_DIR}/lib)
endif()    

include_directories(..)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-D_USE_MATH_DEFINES -DCOMPILE_FOR_ROCM=${COMPILE_FOR_ROCM})

add_executable(${PROJECT_NAME} ${SRC} ${INC})

target_link_libraries(${PROJECT_NAME} hipblaslt hipblas)

set(TARGET_BIN "$<TARGET_FILE_NAME:${PROJECT_NAME}>")

if(NOT ${COMPILE_FOR_ROCM})
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70-real)
    target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})
else()
    set_property(TARGET ${PROJECT_NAME} PROPERTY HIP_ARCHITECTURES gfx90a)
endif()
