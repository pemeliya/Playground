
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

cmake_minimum_required(VERSION 3.12)

project(topk)

set(COMPILE_FOR_ROCM 1)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)

file(GLOB INC *.h *.hpp ../common/*.h ../common/*.hpp)
file(GLOB SRC *.cpp *.cc)
file(GLOB GPU_SRC *.cu.cc)

if(NOT ${COMPILE_FOR_ROCM})
    enable_language(CUDA)
    #set(CMAKE_CUDA_STANDARD 17)
    set(CUDA_64_BIT_DEVICE_CODE ON)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_VERBOSE_BUILD ON)
    #set(CUDA_NVCC_FLAGS "--ptxas-options;--maxrregcount=128;")
    set(CUDA_NVCC_FLAGS "--ptxas-options;-v;-keep")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --ptxas-options=-v -arch=compute_70 -code=sm_70")
    set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE CUDA)
else()
    #list(APPEND CMAKE_PREFIX_PATH "/opt/rocm-5.6.0/lib/cmake/AMDDeviceLibs")
    #list(APPEND CMAKE_PREFIX_PATH "/opt/rocm-5.6.0/lib/cmake/amd_comgr")
    list(APPEND CMAKE_MODULE_PATH "/opt/rocm-5.6.0/lib/cmake/hip") 
    #to be added to hip-lang-config.cmake:
    #find_dependency(AMDDeviceLibs REQUIRED PATHS "/opt/rocm-5.6.0/lib/cmake/AMDDeviceLibs")
    #find_dependency(amd_comgr REQUIRED PATHS "/opt/rocm-5.6.0/lib/cmake/amd_comgr")

    enable_language(HIP) 
    find_package(HIP REQUIRED)
    set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(${SRC} PROPERTIES LANGUAGE HIP)
endif()

# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++20")
include_directories("..")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-D_USE_MATH_DEFINES -DCOMPILE_FOR_ROCM=${COMPILE_FOR_ROCM})

add_executable(${PROJECT_NAME} ${SRC} ${INC})

set(TARGET_BIN "$<TARGET_FILE_NAME:topk>")

if(NOT ${COMPILE_FOR_ROCM})

    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70-real)

# NOTEL: this only worls with 'cubin' not exe files
#add_custom_command(TARGET ${PROJECT_NAME}
#    POST_BUILD
#    COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvdisasm -ndf ${TARGET_BIN}
#    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#)
# add_custom_command(TARGET ${PROJECT_NAME}
#     POST_BUILD
#     COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/cuobjdump -sass ${TARGET_BIN}
#     WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
# )
#add_custom_command(TARGET ${PROJECT_NAME}
#    POST_BUILD
#    COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/cuda-memcheck.exe --print-level info ${TARGET_BIN}
#    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#    COMMENT "=== running cuda-memcheck ==="
#)
    target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})
else()
    set_property(TARGET ${PROJECT_NAME} PROPERTY HIP_ARCHITECTURES gfx90a)
endif()
