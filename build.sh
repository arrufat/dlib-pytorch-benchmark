#!/usr/bin/env bash

# Environment
[ ! -z ${DLIB_USE_CUDA} ] || DLIB_USE_CUDA=ON

# start building
[ -d build ] || mkdir build
cd build
cmake .. -G Ninja -Wno-dev \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DUSE_AVX_INSTRUCTIONS=ON \
  -DUSE_SSE2_INSTRUCTIONS=ON \
  -DUSE_SSE4_INSTRUCTIONS=ON \
  -DDLIB_USE_CUDA=${DLIB_USE_CUDA} \
  ${@}

cmake --build . --config Release
cd -

[[ -f compile_commands.json ]] || ln -s build/compile_commands.json .
