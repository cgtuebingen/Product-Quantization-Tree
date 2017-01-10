cmake_minimum_required(VERSION 2.8)
project(app_project)

set (CMAKE_CXX_COMPILER "g++")
set(EIGEN_PATH "~/libs/eigen")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -DNDEBUG -Wall -Wextra ")

include_directories(SYSTEM ${EIGEN_PATH})

add_executable(build_tree tools/build_tree.cpp)
add_executable(build_db tools/build_db.cpp)
add_executable(query tools/query.cpp)