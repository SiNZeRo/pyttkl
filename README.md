# pyttkl

## Links

- [English Docs](docs/README.md)
- [中文文档](docs/README_zh_cn.md)

## Description

Some Python tools

### kits.py

- This module contains utility functions for handling strings, lists, and other data types.
- Extensions related to logging
- Extensions related to argparse

### tmat.py

- A Trivial Matrix concept implementation that can read and write `pd.DataFrame`.
- Although internally it is stored as numpy-compatible data, considering `pd.DataFrame(ndarray)`, it essentially wraps an ndarray or adds row and column names.
- `load_tmat(filename, mmap_mode='c')`: Automatically determines the payload type based on the meta information in the file and reads it. If the payload is mmap, you can specify `mmap_mode`.
- `save_tmat(filename, tmat, compress='zstd')`: Allows compressing the payload with zstd. There are more options in the code. It can also save as mmap.

## Installation

You can install this package using `pip`:

```bash
pip install .
```

## Design

### tmat format:
- **Design Goals**:
  - 1. Compatible with numpy's ndarray
  - 2. Compatible with pandas' DataFrame
  - 3. Supports compression
  - 4. Supports mmap
  - 5. Supports multiple data types
  - 6. **Native support for both C++ and Python**
  - 7. **Understandable by quantitative researchers by reading the code**
- Example: `TMT\n{header_length: 954}\n{"columns": ["a", "b", "c"], "rows": [1, 2, 3], "dtype": "int32", "compress": "zstd"}\n{payload}`
- Header: 4-byte magic number, `TMT\n`
- JSON string `{"header_length": 954}\n`. If `header_length > 0`, the following meta is a payload compressed with lz4 of length `header_length`. Otherwise, it is a plain JSON string.
  - Trick: Use `head -2 KLINE/OPEN/202401.tmt` to view the file header.
- Meta: JSON string `{"columns": ["a", "b", "c"], "rows": [1, 2, 3], "dtype": "int32", "compress": "zstd"}\n`
- Payload: Binary data. The specific format depends on the meta content.
  - If `{"compress": "zstd"}`, it is zstd-compressed binary data. It can also be lz4 or zstd.
  - Otherwise, it is plain binary data, which can be directly loaded with mmap. You can specify the loading mode with `mmap_mode`.
    - Trick: You can use pandas and numpy's inplace operations to modify the file content directly. Be sure to check `ndarray.flags` when using it.

## TODOS or NO-TODOS
There are many things that can be done, but to keep the project simple and at a level understandable by quantitative researchers, many TODOs become NO TODOs. Of course, if anyone thinks something can be improved, feel free to create an issue.

- TODOs
  - [ ] Code comments
  - [ ] Code documentation
  - [ ] Restrict exposed APIs

- NO TODOs
  - [ ] Multithreading

## Requirements

- Python 3.11 or higher
