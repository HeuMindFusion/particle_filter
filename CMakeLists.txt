CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(DetectLane)


set(CMAKE_CXX_STANDARD 14)


if (NOT ROOT_DIR_DEFINED)
	add_definitions(-DROOT_DIR="${PROJECT_SOURCE_DIR}")
	set(ROOT_DIR_DEFINED ON)
endif()

	

find_package(OpenCV REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})


link_libraries(opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs )


				
ADD_EXECUTABLE(sanyLaneDetectePosition ./main.cpp  ./resource.h)

TARGET_LINK_LIBRARIES(sanyLaneDetectePosition  ${OpenCV_LIBRARIES} )

