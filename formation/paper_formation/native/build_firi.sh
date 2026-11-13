#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
g++ -O3 -std=c++17 -fPIC -shared -I/usr/include/eigen3 -I"$repo_dir/third_party/gcopter_firi" "$repo_dir/formation/paper_formation/native/firi_binding.cpp" -o "$repo_dir/formation/paper_formation/native/libgcopter_firi.so"
