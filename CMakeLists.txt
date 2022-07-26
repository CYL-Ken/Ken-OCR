cmake_minimum_required(VERSION 3.10)

project( CylTekOCR )
find_package( OpenCV REQUIRED )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(
	include
	/usr/include
 	/usr/include/freetype2/
	
)

link_directories(./lib)

file(GLOB CylTekOCR_sources ./src/**.cpp ./src/**.c)
add_executable( CylTekOCR ${CylTekOCR_sources} )
target_link_libraries( CylTekOCR ${OpenCV_LIBS} freetype tesseract jsoncpp curl tensorflow-lite ${CMAKE_DL_LIBS})

