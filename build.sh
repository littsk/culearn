#!/bin/bash

set -ex

# 获取脚本所在的目录
script_dir=$(dirname "$0")

# 切换到项目根目录
cd "$script_dir"

# 创建并进入build目录
rm -rf build
mkdir -p build
cd build

# 运行CMake进行配置和编译
cmake .. -DCMAKE_BUILD_TYPE=Debug
make