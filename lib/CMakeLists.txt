cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(gotorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions(-DGOTORCH_EXPORT)

include_directories(../internal/torch)

add_library(gotorch SHARED
	../internal/torch/api.h
	../internal/torch/loss.h loss.cpp
	../internal/torch/operator.h operator.cpp
	../internal/torch/optimizer.h optimizer.cpp
	../internal/torch/tensor.h tensor.cpp)
target_link_libraries(gotorch "${TORCH_LIBRARIES}")
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	target_link_libraries(gotorch -static-libgcc -static-libstdc++)
endif()
set_property(TARGET gotorch PROPERTY CXX_STANDARD 14)