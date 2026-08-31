cmake_minimum_required(VERSION 3.16)
project(miniinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(MINIINFER_ENABLE_AVX2 "Build AVX2 CPU kernels" OFF)
option(MINIINFER_ENABLE_WEBGPU "Build the native Dawn WebGPU backend" OFF)
option(MINIINFER_BUILD_TESTS "Build tests" ON)
option(MINIINFER_BUILD_WASM "Build the Emscripten browser module" OFF)

find_package(Threads REQUIRED)

add_library(miniinfer
  src/tensor.cpp src/weight.cpp src/kernels.cpp src/tokenizer.cpp src/model.cpp
  src/cpu_backend.cpp src/transformer.cpp src/c_api.cpp)
target_include_directories(miniinfer PUBLIC include)
target_link_libraries(miniinfer PUBLIC Threads::Threads)
if(MINIINFER_ENABLE_AVX2)
  target_compile_definitions(miniinfer PUBLIC MINIINFER_ENABLE_AVX2=1)
  target_sources(miniinfer PRIVATE src/cpu_avx2.cpp)
  if(MSVC)
    set_source_files_properties(src/cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
  else()
    set_source_files_properties(src/cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS -mavx2)
  endif()
endif()

if(MINIINFER_ENABLE_WEBGPU)
  find_package(Dawn REQUIRED)
  target_sources(miniinfer PRIVATE src/webgpu_backend.cpp)
  target_compile_definitions(miniinfer PUBLIC MINIINFER_ENABLE_WEBGPU=1)
  target_compile_features(miniinfer PUBLIC cxx_std_20)
  target_link_libraries(miniinfer PUBLIC dawn::webgpu_dawn)
  add_executable(miniinfer-webgpu-matmul tools/webgpu_matmul.cpp)
  target_link_libraries(miniinfer-webgpu-matmul PRIVATE miniinfer)
endif()

if(NOT EMSCRIPTEN)
  add_executable(infer tools/infer.cpp)
  target_link_libraries(infer PRIVATE miniinfer)
  add_executable(miniinfer-bench tools/bench.cpp)
  target_link_libraries(miniinfer-bench PRIVATE miniinfer)
  add_executable(miniinfer-convert tools/convert.cpp)
  target_link_libraries(miniinfer-convert PRIVATE miniinfer)
  target_compile_definitions(miniinfer-convert PRIVATE
    MINIINFER_CONVERTER_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_hf.py")
endif()

if(MINIINFER_BUILD_WASM)
  if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "MINIINFER_BUILD_WASM requires the Emscripten toolchain")
  endif()
  add_executable(miniinfer-wasm tools/wasm_entry.cpp)
  target_link_libraries(miniinfer-wasm PRIVATE miniinfer)
  set_target_properties(miniinfer-wasm PROPERTIES OUTPUT_NAME miniinfer_wasm SUFFIX ".js")
  target_link_options(miniinfer-wasm PRIVATE
    "SHELL:-sALLOW_MEMORY_GROWTH=1"
    "SHELL:-sALLOW_TABLE_GROWTH=1"
    "SHELL:-sEXPORTED_FUNCTIONS=['_miniinfer_create','_miniinfer_destroy','_miniinfer_generate','_miniinfer_cancel','_miniinfer_tokenize','_miniinfer_decode','_miniinfer_last_error']"
    "SHELL:-sEXPORTED_RUNTIME_METHODS=['ccall','FS','addFunction','removeFunction','UTF8ToString']")
endif()

if(MINIINFER_BUILD_TESTS AND NOT EMSCRIPTEN)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_executable(miniinfer-tests tests/test_core.cpp)
  target_link_libraries(miniinfer-tests PRIVATE miniinfer)
  add_test(NAME miniinfer-tests COMMAND miniinfer-tests)
  add_test(NAME miniinfer-converter-test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_converter.py
            $<TARGET_FILE:miniinfer-convert> $<TARGET_FILE:infer>)
endif()
