
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

cmake_minimum_required(VERSION 3.12)

project(topk)

set(COMPILE_FOR_ROCM 0)

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
    #set(CUDA_BUILD_CUBIN OFF)
    # set(CUDA_NVCC_FLAGS "--compiler-options;-mtune=core2;-mfpmath=sse")
    #set(CUDA_NVCC_FLAGS "--ptxas-options;--maxrregcount=128;")
    set(CUDA_NVCC_FLAGS "--ptxas-options;-v;-keep")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --ptxas-options=-v -arch=compute_70 -code=sm_70")
    set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE CUDA)
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++20")
include_directories("../common")

# clang this will pass '-v' option to ptxas assembler
#set(CUDA_NVCC_FLAGS "-Xcuda-ptxas;-v;-Xcuda-ptxas;--opt-level=3;-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -D_DLL -Xclang --dependent-lib=msvcrt -D_MD")

#cuda_select_nvcc_arch_flags(ARCH_FLAGS 3.5)
#list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS} -Wno-deprecated-gpu-targets)

#cuda_compile_cubin(CUBIN_FILES ${CU})
#cuda_compile(CUDA_FILES ${CU})

set_source_files_properties(TopK_main.cc PROPERTIES LANGUAGE CUDA)

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

endif()

#add_custom_command(TARGET ${PROJECT_NAME}
#    POST_BUILD
#    COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/cuda-memcheck.exe --print-level info ${TARGET_BIN}
#    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#    COMMENT "=== running cuda-memcheck ==="
#)

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})

# Add a special target to clean nvcc generated files.
# CUDA_BUILD_CLEAN_TARGET()
