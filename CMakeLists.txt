cmake_minimum_required(VERSION 2.8)
project( calibration )
find_package( OpenCV REQUIRED )

if (${OpenCV_VERSION_MAJOR} EQUAL 3)
    set(OpenCV_VERSION3 true)
else (${OpenCV_VERSION_MAJOR} EQUAL 3)
    set(OpenCV_VERSION3 false)
endif (${OpenCV_VERSION_MAJOR} EQUAL 3)
#message ("${OpenCV_VERSION} ${OpenCV_VERSION3}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_executable(BinReg BinaryRegistration.cpp ExtractContours.cpp)
target_link_libraries(BinReg ${OpenCV_LIBS})

