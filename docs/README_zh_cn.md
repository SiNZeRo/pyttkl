# pyttkl

## 链接

- [Enligh Docs](docs/README.md)
- [中文文档](docs/README_zh_cn.md)

## 描述

一些 Python 工具

### kits.py

- 该模块包含一些实用程序函数，用于处理字符串、列表和其他数据类型。
- logging 相关扩展
- argparse 相关扩展

### tmat.py

- 一个Trivial Matrix概念实现，可以读写pd.DataFrame
- 虽然内部存储其实只是numpy兼容的。虑到pd.DataFrame(ndarray)，其实只是把ndarray wrapper了下。或者加了行名和列名。
- `load_tmat(filename, mmap_mode='c')`, 会自动根据文件里的meta信息判断文件的payload类型进行读取，额外的如果payload是mmap的话，可以指定mmap_mode
- `save_tmat(filename, tmat, compress='zstd')`, 可以用zstd压缩payload, 还有更多选项看代码吧。也可以存成mmap

## Design

### tmat的格式:
- example: `TMT\n{header_length: 954}\n{"columns": ["a", "b", "c"], "rows": [1, 2, 3], "dtype": "int32", "compress": "zstd"}\n{payload}`
- 头部: 4字节的magic number, `TMT\n`
- json字符串 `{"header_length": 954}\n`, 如果header_length > 0， 则后面的meta事用lz4压缩的payload长度为`header_length`, 某则就是明文的json字符串
   - trick, 可以用 `head -2 KLINE/OPEN/202401.tmt` 来查看文件的头部
- meta: json字符串 `{"columns": ["a", "b", "c"], "rows": [1, 2, 3], "dtype": "int32", "compress": "zstd"}\n`,
- payload: 二进制数据, 具体格式取决于meta的内容
  - 如果是`{"compress": "zstd"}`, 则是zstd压缩的二进制数据, 可以是lz4, zstd
  - 否则就是明文的二进制数据, 直接用mmap加载, 加载的时候可以用`mmap_mode`指定加载模式
    - trick: 可以使用pandas和numpy的inplace操作直接修改文件内容. 当然用的时候记得看 `ndarray.flags`

## TODOS or NOTODES
就很多事可以做，但是为了保证项目比较简单，大约就是量化交易的研究员可以看的懂的水平，很多TODO就变成了 NO TODO. 当然有兄弟们觉得哪些可以搞可以建issue

- TODOs
  - [ ] 代码注释
  - [ ] 代码文档
  - [ ] 限制下对外开放的api

- NO TODOs
  - [ ] 多线程

## 要求
- Python 3.11 或更高版本