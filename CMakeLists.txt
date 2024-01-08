cmake_minimum_required(VERSION 3.0.2)
project(pcl_subscriber)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_language(CUDA)

# Get the path to the root of the source tree
get_filename_component(centerpoint_DIR "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)

# set centerpoint include path
set(centerpoint_INCLUDE_DIRS ${centerpoint_DIR}/include)
set(TENSORRT_DIRS /home/TensorRT-8.0.1.6)
set(CUDA_LIB_PATH /usr/local/cuda/lib64/)


find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  rospy
  sensor_msgs
)

# pcl and boost related !
find_package(Boost COMPONENTS program_options REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
 
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# set flags for TensorRT availability
option(TRT_AVAIL "TensorRT available" OFF)
# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

if(NVINFER AND NVPARSERS AND NVONNXPARSERS)
  message("TensorRT is available!")
  message("NVINFER: ${NVINFER}")
  message("NVPARSERS: ${NVPARSERS}")
  message("NVONNXPARSERS: ${NVONNXPARSERS}")
  set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

file(GLOB PointPillarLibs ${centerpoint_DIR}/src/*cpp)
add_library(pointpillars SHARED ${PointPillarLibs})

file(GLOB PointPillarCU ${centerpoint_DIR}/src/*cu)
add_library(pointpillarsCU SHARED ${PointPillarCU})

catkin_package()


# set flags for CUDA availability
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
message("CUDA dir paths", ${CUDA_LIBRARIES})


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${centerpoint_INCLUDE_DIRS}
  ${centerpoint_INCLUDE_DIRS}/common
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_DIRS}/include
  include
)

add_executable(${PROJECT_NAME}_node 
  src/pcl_sub_node.cpp
  src/trtCenterPoint.cpp
  ${centerpoint_DIR}/include/common/logger.cpp
)


target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARY}
  ${CUDA_RUNTIME_LIBRARY}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CUDA_LIB_PATH}libcudart.so
  ${TENSORRT_DIRS}/lib/libnvinfer.so
  ${TENSORRT_DIRS}/lib/libnvonnxparser.so
  pointpillars
  pointpillarsCU
)
