import logging
import os
import sys
import multiprocessing
from datetime import datetime, timedelta
import time
import json
import zstandard as zstd
import lz4.frame
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compress_lz4(data):
    return lz4.frame.compress(data)

def decompress_lz4(data):
    return lz4.frame.decompress(data)

def compress_zstd(data):
    cctx = zstd.ZstdCompressor()
    return cctx.compress(data)

def decompress_zstd(data):
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)

def str_to_dtype(dtype: str) -> np.dtype:
    if dtype == 'float32':
        return np.float32
    elif dtype == 'float64':
        return np.float64
    elif dtype == 'int32':
        return np.int32
    elif dtype == 'int64':
        return np.int64
    elif dtype == 'int16':
        return np.int16
    elif dtype == 'int8':
        return np.int8
    else:
        raise ValueError(f'Invalid dtype {dtype}')

def read_tmat(file_path: str) -> pd.DataFrame:
    '''
    Read a matrix from a file
    '''
    fin = open(file_path, 'rb')

    magic = fin.readline()
    if magic != b'TMT\n':
        raise ValueError('Invalid matrix file')
    js_str = fin.readline().decode('utf-8')
    try:
        js = json.loads(js_str)
    except json.JSONDecodeError:
        raise ValueError('Invalid json string')

    if 'columns' not in js or 'rows' not in js:
        raise ValueError('missing columns or rows in json')

    data = fin.read()

    compress = js.get('compress', False)
    dtype = js.get('dtype', 'float64')
    dtype = str_to_dtype(dtype)

    if compress:
        if compress == 'zstd':
            data = decompress_zstd(data)
        elif compress == 'lz4':
            data = decompress_lz4(data)
        else:
            raise ValueError(f'Invalid compress method {compress}')

    columns = js['columns']
    rows = js['rows']

    npv = np.frombuffer(data, dtype=dtype).reshape(len(rows), len(columns))

    return pd.DataFrame(npv, columns=columns, index=rows)

def write_tmat(file_path: str,
               df: pd.DataFrame,
               compress: str = 'zstd',
               dtype: str = 'float64'):
    '''
    Write a matrix to a file
    '''
    fout = open(file_path, 'wb')

    columns = df.columns
    rows = df.index.tolist()

    js = {
        'columns': columns.tolist(),
        'rows': rows,
        'compress': compress,
        'dtype': dtype
    }

    js_str = json.dumps(js)
    fout.write(b'TMT\n')
    fout.write(js_str.encode('utf-8'))
    fout.write(b'\n')

    data = df.astype(dtype).values.tobytes()

    if compress:
        if compress == 'zstd':
            data = compress_zstd(data)
        elif compress == 'lz4':
            data = compress_lz4(data)
        else:
            raise ValueError(f'Invalid compress method {compress}')

    fout.write(data)
    fout.close()


def test_read_write():
    ofn = '/tmp/test_tmat'
    df = pd.DataFrame(np.random.rand(100, 100), columns=[f'col{i}' for i in range(100)])
    df.index = [f'row{i}' for i in range(100)]
    write_tmat(ofn, df, dtype='float32', compress='zstd')
    df2 = read_tmat(ofn)
    print(df)
    print(df2)


def main():
    from pyttkl import kits

    kits.init_logging('debug')
    args = {'__subcmd__': []}
    args.update(kits.make_sub_cmd(write_tmat))
    args.update(kits.make_sub_cmd(read_tmat))
    args.update(kits.make_sub_cmd(test_read_write))
    args = kits.make_args(args)
    kits.run_cmds(args)


if __name__ == "__main__":
    main()
