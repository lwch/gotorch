# gotorch

[![gotorch](https://github.com/lwch/gotorch/actions/workflows/cpu.yml/badge.svg)](https://github.com/lwch/gotorch/actions/workflows/cpu.yml)
[![gotorch](https://github.com/lwch/gotorch/actions/workflows/gpu.yml/badge.svg)](https://github.com/lwch/gotorch/actions/workflows/gpu.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/lwch/gotorch.svg)](https://pkg.go.dev/github.com/lwch/gotorch)

这是一个GO版本的libtorch封装库，通过该库可快速搭建torch的模型，目前已支持最新版本的libtorch(2.0.1)，支持的操作系统如下

- windows
- linux
- macos

## 安装

1. 下载[libtorch](https://pytorch.org/get-started/locally/)，windows下解压到D盘，linux和mac下解压到/usr/local/lib目录下
2. 下载[libgotorch](https://github.com/lwch/gotorch/releases/latest)并放置在libtorch的lib目录下，windows系统使用dll，linux系统使用so(请选择正确的glibc版本并将其更名为libgotorch.so)，macos系统使用dylib

注：由于官方提供的windows版本libtorch使用msvc进行编译，通过mingw无法正常链接，因此增加libgotorch库来进行转换，有关libgotorch库的编译请看[libgotorch编译](docs/libgotorch.md)，另外也可参考[release.yml](.github/workflows/release.yml)中的命令。

### linux

在.bashrc中添加以下内容

```
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/lib/libtorch/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/libtorch/lib"
```

### macos

在.bashrc中添加以下内容

```
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/lib/libtorch/lib"
export DYLD_FALLBACK_LIBRARY_PATH="$DYLD_FALLBACK_LIBRARY_PATH:/usr/local/lib/libtorch/lib"
```

### windows

windows系统下使用cgo需要依赖mingw，推荐使用[llvm-mingw](https://github.com/mstorsjo/llvm-mingw)，并添加以下环境变量

```
LIBRARY_PATH="D:\libtorch\lib"
Path="D:\libtorch\lib;<mingw所在路径>\bin"
```

## 使用

可查看[mlp](example/mlp)中的示例

注意：由于cgo中创建的对象无法被go GC所捕获并释放，因此在实际使用过程中需要通过[mmgr](mmgr)库来捕获创建的tensor对象并通过GC接口来手动释放内存，在From系列接口中生成tensor对象允许给定空的storage对象，该对象一般被用来作为模型参数，因此不会被GC所释放。运算过程中产生的新tensor对象会继承自他的子集的storage，因此在运算工程中生成的临时对象可被storage所捕获并释放。

## feature

- 支持GPU