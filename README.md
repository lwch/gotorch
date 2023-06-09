# gotorch

这是一个GO版本的libtorch封装库，通过该库可快速搭建torch的模型，目前已支持最新版本的libtorch(2.0.1)，支持的操作系统如下

- linux
- macos

## 安装

下载并解压[libtorch](https://pytorch.org/get-started/locally/)

### linux

在.bashrc中添加以下内容

```
export GOTORCH="此处修改为解压路径如：/usr/local/lib/libtorch"
export LIBRARY_PATH="$LIBRARY_PATH:$GOTORCH/lib"
export CPATH="$CPATH:$GOTORCH/lib:$GOTORCH/include:$GOTORCH/include/torch/csrc/api/include"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTORCH/lib"
```

### macos

在.bashrc中添加以下内容

```
export GOTORCH="此处修改为解压路径如：/usr/local/lib/libtorch"
export LIBRARY_PATH="$LIBRARY_PATH:$GOTORCH/lib"
export CPATH="$CPATH:$GOTORCH/lib:$GOTORCH/include:$GOTORCH/include/torch/csrc/api/include"
export DYLD_FALLBACK_LIBRARY_PATH="$DYLD_FALLBACK_LIBRARY_PATH:$GOTORCH/lib"
```

## 使用

可查看[mlp](example/mlp)中的示例

注意：由于cgo中创建的对象无法被go GC所捕获并释放，因此在实际使用过程中需要通过[mmgr](mmgr)库来捕获创建的tensor对象并通过GC接口来手动释放内存，在From系列接口中生成tensor对象允许给定空的storage对象，该对象一般被用来作为模型参数，因此不会被GC所释放。运算过程中产生的新tensor对象会继承自他的子集的storage，因此在运算工程中生成的临时对象可被storage所捕获并释放。

## feature

- 支持GPU