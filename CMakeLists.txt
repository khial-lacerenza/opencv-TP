cmake_minimum_required(VERSION 3.8)
project( DisplayImage )
set (CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

SET( MAIN_EXEC
     tp1)

FOREACH(FILE ${MAIN_EXEC})
            add_executable( ${FILE} ${FILE}.cpp )
            target_link_libraries( ${FILE} ${OpenCV_LIBS} )
ENDFOREACH(FILE)
