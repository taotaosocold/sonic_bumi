# FindONNXRuntime.cmake
# Find ONNX Runtime installation
#
# This module defines:
#  onnxruntime_FOUND - True if ONNX Runtime is found
#  onnxruntime_INCLUDE_DIRS - Include directories for ONNX Runtime
#  onnxruntime_LIBRARIES - Libraries for ONNX Runtime
#  onnxruntime::onnxruntime - Imported target for ONNX Runtime

# G1 和 BUMI3 共用环境中的 C++ SDK；自定义安装根目录优先于系统默认位置。
find_path(onnxruntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${onnxruntime_ROOT} $ENV{onnxruntime_ROOT} $ENV{ONNX_INSTALL_PATH}
  PATHS
    ENV onnxruntime_ROOT
    ENV onnxruntime_DIR
    /opt/onnxruntime
    /usr/local
    /usr
  PATH_SUFFIXES include include/onnxruntime
)

find_library(onnxruntime_LIBRARY
  NAMES onnxruntime
  HINTS ${onnxruntime_ROOT} $ENV{onnxruntime_ROOT} $ENV{ONNX_INSTALL_PATH}
  PATHS
    ENV onnxruntime_ROOT
    ENV onnxruntime_DIR
    /opt/onnxruntime
    /usr/local
    /usr
  PATH_SUFFIXES lib lib64
)

# 官方 SDK 包带有 VERSION_NUMBER；其他安装形式使用头文件 API 版本估计最低版本。
# 版本信息供 BUMI3 拒绝过旧的环境；G1 未指定最低版本，保持原来的查找调用方式。
unset(onnxruntime_VERSION)
if(onnxruntime_INCLUDE_DIR)
  get_filename_component(onnxruntime_SDK_ROOT "${onnxruntime_INCLUDE_DIR}" DIRECTORY)
  if(EXISTS "${onnxruntime_SDK_ROOT}/VERSION_NUMBER")
    file(STRINGS "${onnxruntime_SDK_ROOT}/VERSION_NUMBER" onnxruntime_VERSION LIMIT_COUNT 1)
    string(STRIP "${onnxruntime_VERSION}" onnxruntime_VERSION)
  elseif(EXISTS "${onnxruntime_INCLUDE_DIR}/onnxruntime_c_api.h")
    file(STRINGS "${onnxruntime_INCLUDE_DIR}/onnxruntime_c_api.h" onnxruntime_API_LINE
      REGEX "^#define ORT_API_VERSION [0-9]+" LIMIT_COUNT 1)
    if(onnxruntime_API_LINE MATCHES "ORT_API_VERSION ([0-9]+)")
      set(onnxruntime_VERSION "1.${CMAKE_MATCH_1}.0")
    endif()
  endif()
endif()

# 头文件与动态库都存在且版本满足调用方要求时，才暴露统一的导入目标。
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(onnxruntime
  FOUND_VAR onnxruntime_FOUND
  REQUIRED_VARS onnxruntime_LIBRARY onnxruntime_INCLUDE_DIR
  VERSION_VAR onnxruntime_VERSION
)

if(onnxruntime_FOUND)
  set(onnxruntime_INCLUDE_DIRS ${onnxruntime_INCLUDE_DIR})
  set(onnxruntime_LIBRARIES ${onnxruntime_LIBRARY})

  # Create imported target
  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}"
    )
  endif()

  mark_as_advanced(onnxruntime_INCLUDE_DIR onnxruntime_LIBRARY)
endif()

