


cmake_minimum_required(VERSION 3.12)

project(QCCL)

file(GLOB INC *.h *.hpp ../common/*.h ../common/*.hpp)
file(GLOB SRC *.cpp *.cc)
file(GLOB GPU_SRC *.cu.cc)

if(NOT ${COMPILE_FOR_ROCM})
    set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE CUDA)
else()
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

add_definitions(-D_USE_MATH_DEFINES -DCOMPILE_FOR_ROCM=${COMPILE_FOR_ROCM})

add_library(${PROJECT_NAME} SHARED ${SRC} ${INC})

set(TARGET_BIN "$<TARGET_FILE_NAME:${PROJECT_NAME}>")

if(NOT ${COMPILE_FOR_ROCM})

    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 70-real)
    target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})
else()
    set_property(TARGET ${PROJECT_NAME} PROPERTY HIP_ARCHITECTURES gfx942)
    target_link_libraries(${PROJECT_NAME} rccl pthread rocprofiler64v2)
endif()
