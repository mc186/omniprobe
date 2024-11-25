include(FetchContent)

function(fetch_project)
  cmake_parse_arguments(FETCH_SOURCE
    "INCLUDE_DIRS;EXCLUDE_EXAMPLES"   # options
    "NAME;GIT_REPOSITORY;GIT_TAG" # single-arg options
    ""             # multi-arg options
    ${ARGN}
  )

  set(SOURCE ${CMAKE_CURRENT_BINARY_DIR}/deps/${FETCH_SOURCE_NAME}/source)
  set(BUILD ${CMAKE_CURRENT_BINARY_DIR}/deps/${FETCH_SOURCE_NAME}/build)
  set(SUBBUILD ${CMAKE_CURRENT_BINARY_DIR}/deps/${FETCH_SOURCE_NAME}/subbuild)

#   FetchContent_Populate(${FETCH_SOURCE_NAME}
#     URL ${FETCH_SOURCE_URL}
#     DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/.anari_deps/${FETCH_SOURCE_NAME}
#     ${FETCH_SOURCE_MD5_COMMAND}
#     SOURCE_DIR ${SOURCE}
#   )

  if(FETCH_SOURCE_EXCLUDE_EXAMPLES)
    set(FETCH_SOURCE_EXCLUDE_EXAMPLES ${SOURCE}/examples)
    message(STATUS "Excluding examples from ${FETCH_SOURCE_EXCLUDE_EXAMPLES}")
  endif()

  FetchContent_Declare(${FETCH_SOURCE_NAME}
    GIT_REPOSITORY ${FETCH_SOURCE_GIT_REPOSITORY}
    GIT_TAG ${FETCH_SOURCE_GIT_TAG}
    GIT_SHALLOW ON
    SOURCE_DIR ${SOURCE}
    BINARY_DIR ${BUILD}
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  )
  FetchContent_MakeAvailable(${FETCH_SOURCE_NAME})

  set("${FETCH_SOURCE_NAME}_LOCATION" ${CMAKE_CURRENT_BINARY_DIR}/deps/${FETCH_SOURCE_NAME} PARENT_SCOPE)
  message(STATUS "Fetched ${FETCH_SOURCE_NAME}")

  if (FETCH_SOURCE_INCLUDE_DIRS)
    message(STATUS "Including directory ${SOURCE}")
    target_include_directories(
      ${FETCH_SOURCE_NAME}
      PUBLIC
      ${SOURCE}/include
    )
  endif()
endfunction()