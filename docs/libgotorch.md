# libgotorch编译

由于管饭给出的windows版本的libtorch库使用msvc编译器，因此使用mingw编译器无法正常链接，因此在windows下编译libgotorch库需要先安装vs2019以及c++编译器。

## windows系统

1. 安装cmake
2. 在lib目录下创建build目录
3. 使用以下命令生成vs项目
    ```
    cmake -DCMAKE_PREFIX_PATH=D:\libtorch ..
    ```
4. 打开sln文件并进行编译

## linux和macos系统

1. 安装cmake
2. 在lib目录下创建build目录
3. 使用以下命令生成Makefile并编译
    ```
    cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/libtorch -DCMAKE_BUILD_TYPE=Release ..
    make
    ```