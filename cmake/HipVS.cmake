set(OB_BUILD_CUVS AUTO CACHE STRING "Build the hipVS backend (AUTO, ON, or OFF)")
set_property(CACHE OB_BUILD_CUVS PROPERTY STRINGS AUTO ON OFF)
set(CUVS_BRIDGE_LIB "" CACHE FILEPATH "Existing seekdb hipVS bridge library")

set(hipvs_supported OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(hipvs_supported ON)
endif()

set(hipvs_root "$ENV{HIPVS_ROOT}")
set(rocm_root "$ENV{ROCM_ROOT}")
if(NOT hipvs_root)
  set(hipvs_root /opt/hipvs)
endif()
if(NOT rocm_root)
  set(rocm_root /opt/rocm)
endif()
set(hipvs_compiler "$ENV{HIPCC}")
if(NOT hipvs_compiler)
  set(hipvs_compiler "${rocm_root}/bin/hipcc")
endif()
set(hipvs_sdk_available OFF)
if(EXISTS "${hipvs_compiler}" AND
   EXISTS "${hipvs_root}/include/cuvs/core/c_api.h" AND
   EXISTS "${hipvs_root}/include/cuvs/neighbors/cagra.h" AND
   EXISTS "${hipvs_root}/lib/libcuvs_c.so")
  set(hipvs_sdk_available ON)
endif()

if(OB_BUILD_CUVS STREQUAL "AUTO")
  set(OB_BUILD_CUVS OFF)
  if(hipvs_supported AND NOT CMAKE_CROSSCOMPILING AND NOT OB_BUILD_PACKAGE AND
     (CUVS_BRIDGE_LIB OR hipvs_sdk_available))
    set(OB_BUILD_CUVS ON)
  endif()
endif()

if(OB_BUILD_CUVS)
  if(NOT hipvs_supported)
    message(FATAL_ERROR "OB_BUILD_CUVS requires Linux x86_64")
  endif()
  if(CUVS_BRIDGE_LIB)
    get_filename_component(hipvs_bridge_library "${CUVS_BRIDGE_LIB}" ABSOLUTE)
    if(NOT EXISTS "${hipvs_bridge_library}" OR IS_DIRECTORY "${hipvs_bridge_library}")
      message(FATAL_ERROR "CUVS_BRIDGE_LIB is not a library file: ${hipvs_bridge_library}")
    endif()
  elseif(hipvs_sdk_available)
    set(hipvs_bridge_source "${CMAKE_SOURCE_DIR}/tools/hipvs_bridge")
    set(hipvs_bridge_library "${CMAKE_BINARY_DIR}/hipvs/libseekdb_cuvs_bridge.so")
    add_custom_command(
      OUTPUT "${hipvs_bridge_library}"
      COMMAND "${CMAKE_COMMAND}" -E env
        "HIPVS_ROOT=${hipvs_root}" "ROCM_ROOT=${rocm_root}" "HIPCC=${hipvs_compiler}"
        bash "${hipvs_bridge_source}/build.sh" "${hipvs_bridge_library}"
      DEPENDS
        "${hipvs_bridge_source}/build.sh"
        "${hipvs_bridge_source}/seekdb_cuvs_bridge.c"
        "${hipvs_bridge_source}/seekdb_cuvs_bridge.h"
        "${hipvs_root}/lib/libcuvs_c.so"
      COMMENT "Building the seekdb hipVS bridge"
      VERBATIM)
    add_custom_target(seekdb_hipvs_bridge_build DEPENDS "${hipvs_bridge_library}")
  else()
    message(FATAL_ERROR
      "OB_BUILD_CUVS=ON requires the hipVS and ROCm SDKs or an existing CUVS_BRIDGE_LIB. "
      "Check HIPVS_ROOT=${hipvs_root} and HIPCC=${hipvs_compiler}.")
  endif()

  add_library(seekdb_hipvs_bridge SHARED IMPORTED GLOBAL)
  set_target_properties(seekdb_hipvs_bridge PROPERTIES IMPORTED_LOCATION "${hipvs_bridge_library}")
  if(TARGET seekdb_hipvs_bridge_build)
    add_dependencies(seekdb_hipvs_bridge seekdb_hipvs_bridge_build)
  endif()
  add_definitions(-DOB_BUILD_CUVS)
  message(STATUS "hipVS enabled: ${hipvs_bridge_library}")
endif()
