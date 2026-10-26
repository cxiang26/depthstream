#!/bin/bash
# ORB_SLAM3 全链编译脚本(DBoW2 -> g2o -> Sophus -> 词库 -> 主工程)
# 用法: nohup bash ~/orb_build_chain.sh > /tmp/orb_chain.log 2>&1 &
set -x
cd ~/ORB_SLAM3

# 放宽 OpenCV 版本门槛(实际兼容 4.1)
grep -q 'find_package(OpenCV 4)' CMakeLists.txt || \
    sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4)/' CMakeLists.txt

for TP in DBoW2 g2o Sophus; do
    cd ~/ORB_SLAM3/Thirdparty/$TP
    rm -rf build && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release > /tmp/${TP}.log 2>&1 || exit 1
    make -j2 >> /tmp/${TP}.log 2>&1 || exit 1
    echo "=== $TP DONE ==="
done

cd ~/ORB_SLAM3/Vocabulary
[ -f ORBvoc.txt ] || tar -xf ORBvoc.txt.tar.gz

cd ~/ORB_SLAM3
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release > /tmp/orbslam_cmake.log 2>&1 || exit 1
make -j2 > /tmp/orbslam_make.log 2>&1 || exit 1
echo ORB_BUILD_OK > /tmp/orb_build.status
