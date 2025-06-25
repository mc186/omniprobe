################################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

include(ExternalProject)

function(ext_proj_add)
    cmake_parse_arguments(EXT_PROJ_ADD 
        "INCLUDE_DIRS" #options
        "NAME;C_COMPILER;CXX_COMPILER;BINARY_SUFFIX" # single-arg options
        "" # multi-arg options
        ${ARGN}
    )
    set (EXTERNAL_PROJ_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/${EXT_PROJ_ADD_NAME})
    
    # Optionally include the include files
    if (EXT_PROJ_ADD_INCLUDE_DIRS)
        include_directories(${EXTERNAL_PROJ_DIR}/include)
        message(STATUS "Including directory: ${EXTERNAL_PROJ_DIR}/include")
    endif()
    
    # Add project as a subdirectory, optionally set custom compiler
    if (EXT_PROJ_ADD_C_COMPILER AND EXT_PROJ_ADD_CXX_COMPILER) 
        ExternalProject_Add(
            ${EXT_PROJ_ADD_NAME}${EXT_PROJ_ADD_BINARY_SUFFIX}
            PREFIX ${CMAKE_BINARY_DIR}/external/${EXT_PROJ_ADD_NAME}${EXT_PROJ_ADD_BINARY_SUFFIX}
            SOURCE_DIR ${EXTERNAL_PROJ_DIR}
            BINARY_DIR ${CMAKE_BINARY_DIR}/external/${EXT_PROJ_ADD_NAME}${EXT_PROJ_ADD_BINARY_SUFFIX}/build
            CMAKE_ARGS 
            -DCMAKE_C_COMPILER=${EXT_PROJ_ADD_C_COMPILER} 
            -DCMAKE_CXX_COMPILER=${EXT_PROJ_ADD_CXX_COMPILER} 
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
            $<$<BOOL:${LLVM_INSTALL_DIR}>:-DLLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}>
            BUILD_ALWAYS ON   
        )
    else()
        add_subdirectory(${EXTERNAL_PROJ_DIR})
    endif()
    
    # Track include and lib directories
    string(TOUPPER "${EXT_PROJ_ADD_NAME}" CAPS_NAME)
    set("${CAPS_NAME}_INCLUDE_DIR" ${EXTERNAL_PROJ_DIR}/include PARENT_SCOPE)
    set("${CAPS_NAME}_LIBRARIES"  ${CMAKE_CURRENT_BINARY_DIR}/external/${EXT_PROJ_ADD_NAME}/lib PARENT_SCOPE)
    
endfunction()
