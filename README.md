# gotorch

[![gotorch](https://github.com/lwch/gotorch/actions/workflows/cpu.yml/badge.svg)](https://github.com/lwch/gotorch/actions/workflows/cpu.yml)
[![gotorch](https://github.com/lwch/gotorch/actions/workflows/gpu.yml/badge.svg)](https://github.com/lwch/gotorch/actions/workflows/gpu.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/lwch/gotorch.svg)](https://pkg.go.dev/github.com/lwch/gotorch)

这是一个GO版本的libtorch封装库，通过该库可快速搭建torch的模型，目前已支持最新版本的libtorch(2.0.1)，支持的操作系统如下

- windows
- linux
- macos

已支持*CPU*和*GPU*运算

## 安装

1. 下载[libtorch](https://pytorch.org/get-started/locally/)，windows下解压到D盘，linux和mac下解压到/usr/local/lib目录下
2. 下载[libgotorch](https://github.com/lwch/gotorch/releases/latest)并放置在libtorch的lib目录下
  - windows操作系统请更名为gotorch.dll
  - linux操作系统请根据glibc版本下载对应so文件并更名为libgotorch.so
  - macos最新版本仅支持arm64架构，下载后请更名为libgotorch.dylib

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

```go
a := tensor.ARange(nil, 6, consts.KFloat,
    tensor.WithShape(2, 3),
    tensor.WithDevice(consts.KCUDA))
b := tensor.ARange(nil, 6, consts.KFloat,
    tensor.WithShape(3, 2),
    tensor.WithDevice(consts.KCUDA))
c := a.MatMul(b)
fmt.Println(c.ToDevice(consts.KCPU).Float32Value()) // 注意：显存中的数据无法直接读取，需将其转换到CPU后才可读取
```

**注意: 由于大部分tensor对象在C栈中创建，在go中无法正确捕获内存用量，因此建议在长期运行的服务中（如模型训练）使用debug.SetGCPercent将go的GC关闭并在每个迭代中手动调用runtime.GC进行内存释放**

## 模型的checkpoint加载

```go
m, _ := model.Load("yolo_tiny.pt", nil)
for name, t := m.Params() {
    fmt.Println(name, t.Shapes())
}
```

## 版本维护

| gotorch版本 | libtorch版本 |
| --- | --- |
| v1.0.0~v1.5.7 | v2.0.1 |
| v1.6.0~v1.7.2 | v2.1~v2.2.1 |
| v1.7.3 | v2.2.2 |
| v1.7.4~v1.8.0 | v2.3.1 |
| v1.9.0 | v2.4.0 |