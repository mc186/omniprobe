################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set ( TARGET_LIB "${TARGET_NAME}" )
set ( LIB_SRC
  ${LIB_DIR}/interceptor.cc
  ${LIB_DIR}/utils.cc
  ${LIB_DIR}/hsa_mem_mgr.cc
  ${LIB_DIR}/comms_mgr.cc
  ${LIB_DIR}/pyHandler.cc
  ${LIB_DIR}/memory_heatmap.cc
)

find_package(Python COMPONENTS Development)

function(find_file_directory FILENAME DIRS OUTPUT_VAR)
    # Search for the file in the provided directories
    find_file(FILE_FOUND ${FILENAME} PATHS ${DIRS})

    # Check if the file was found
    if(FILE_FOUND)
        # Extract the directory from the found file path
        get_filename_component(DIR_FOUND ${FILE_FOUND} DIRECTORY)
        # Set the output variable to the found directory
        set(${OUTPUT_VAR} ${DIR_FOUND} PARENT_SCOPE)
    else()
        # If the file is not found, set the output variable to an empty value
        set(${OUTPUT_VAR} "" PARENT_SCOPE)
        message(WARNING "File ${FILENAME} not found in directories: ${DIRS}")
    endif()
endfunction()

set(DH_COMMS_SEARCH_DIRS
    ${CMAKE_INCLUDE_PATH}
    )

find_file_directory("dh_comms.h" "${DH_COMMS_SEARCH_DIRS}"  DH_COMMS_INCLUDE)

if(DH_COMMS_INCLUDE)
    message(STATUS "Found dh_comms include directory: ${DH_COMMS_INCLUDE}")
else()
    message(WARNING "Could not find dh_comms include directory")
    message(WARNING "This project has a dependency on the dh_comms library which can be found here: https://github.com/AARInternal/dh_comms.git")
endif()

find_library(DH_COMMS_LIB NAMES dh_ omms libdh_comms.so)

if(DH_COMMS_LIB)
    message(STATUS "Found dh_comms library: ${DH_COMMS_LIB}")
else()
    message(WARNING "Could not find dh_comms library")
    message(WARNING "This project has a dependency on the dh_comms library which can be found here: https://github.com/AARInternal/dh_comms.git")
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
    separate_arguments(
        CLANG_ARG_LIST 
        UNIX_COMMAND
        "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
    message("generate_hsaco: ${PROJECT_BINARY_DIR} ${INPUT_FILE} ${TARGET_ID}")
    add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/test_kernels/${OUTPUT_FILE}
        COMMAND ${clang} ${CLANG_ARG_LIST}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/test_kernels/
        DEPENDS ${INPUT_FILE}
        COMMENT
            "Building ${OUTPUT_FILE}..." -E copy
            ${PROJECT_BINARY_DIR}/test_kernels/${OUTPUT_FILE}
            ${PROJECT_BINARY_DIR}/test_kernels/${target_id}_copy.hsaco
        VERBATIM)
    set(HSACO_TARGET_LIST
        ${HSACO_TARGET_LIST}
        ${PROJECT_BINARY_DIR}/test_kernels/${OUTPUT_FILE}
        PARENT_SCOPE)        
endfunction(generate_hsaco)

set(GPU_LIST "gfx900" "gfx906" "gfx908" "gfx90a" "gfx942" "gfx1030" "gfx1031" "gfx1032" "gfx1100" "gfx1101" "gfx1102")
# Create test_kernels directory if it doesn't exist
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/test_kernels)
# Run generate_hsaco for each target
foreach(target_id ${GPU_LIST})
    # generate kernel bitcodes
    message("${CMAKE_CURRENT_SOURCE_DIR}")
    generate_hsaco(${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/src/test_kernels/copy.cl
                   ${target_id}_copy.hsaco)
    install(
        FILES
            "${PROJECT_BINARY_DIR}/test_kernels/${target_id}_copy.hsaco"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/test_kernels"
        COMPONENT tests)

endforeach(target_id)
message("${HSACO_TARGET_LIST}")
add_custom_target(hsaco_targets DEPENDS ${HSACO_TARGET_LIST})


link_directories(${ROCM_ROOT_DIR}/lib $ENV{HOME}/.local/lib64)
add_library ( ${TARGET_LIB} SHARED ${LIB_SRC})
add_dependencies(${TARGET_LIB} hsaco_targets)

target_include_directories (
    ${TARGET_LIB}
  PRIVATE
    ${LIB_DIR}
    ${ROCM_ROOT_DIR}/include
    ${ROOT_DIR}
    ${HSA_RUNTIME_INC_PATH}
    ${HSA_KMT_LIB_PATH}/..
    ${DH_COMMS_INCLUDE}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(
    ${TARGET_LIB}
  PRIVATE
    ${HSA_RUNTIME_LIB}
    c
    stdc++
    stdc++fs
    ${DH_COMMS_LIB}
    amd_comgr
)



