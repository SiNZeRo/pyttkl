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
from pyttkl.kits import logging_trace

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
    return dctx.decompress(data, max_output_size=24 * 1024 * 1024 * 1024)


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


def read_tmat(file_path: str, mmap_mode='c') -> pd.DataFrame:
    '''
    Read a matrix from a file

    '''
    fin = open(file_path, 'rb')

    magic = fin.readline()
    if magic != b'TMT\n':
        raise ValueError('Invalid matrix file')

    # read the json string
    params_js = fin.readline().decode('utf-8')
    try:
        params = json.loads(params_js)
    except json.JSONDecodeError as e:
        logger.error(f'Failed to decode json: {e}')
        raise e

    compress_header = params.get('header_length', 0)
    compress = params.get('compress', None)
    if compress is None:
        raise ValueError('missing compress in header')

    if compress_header > 0:
        js_str = fin.read(compress_header)
        js_str = decompress_lz4(js_str)
    else:
        js_str = fin.readline().decode('utf-8')

    try:
        js = json.loads(js_str)
    except json.JSONDecodeError as e:
        logger.error(f'Failed to decode json: {e}')
        raise e

    if 'columns' not in js or 'rows' not in js:
        raise ValueError('missing columns or rows in json')

    dtype = js.get('dtype', 'float64')
    dtype = str_to_dtype(dtype)

    logging_trace(logger, f'js {js}')

    if compress != 'mmap':
        data = fin.read()
        if fin is not None:
            fin.close()
        if compress == 'zstd':
            data = decompress_zstd(data)
        elif compress == 'lz4':
            data = decompress_lz4(data)
        else:
            raise ValueError(f'Invalid compress method {compress}')
    else:
        try:
            memmap = np.memmap(file_path, dtype=dtype, mode=mmap_mode, offset=fin.tell())
        except Exception as e:
            logger.error(f'Failed to create memmap: {e}')
            raise e

        logging_trace(
            logger, f'loaded memmap {file_path} {dtype} {mmap_mode} {fin.tell()}')

        if fin is not None:
            fin.close()
        data = memmap

    columns = js['columns']
    rows = js['rows']

    npv = np.frombuffer(data, dtype=dtype).reshape(len(rows), len(columns))
    return pd.DataFrame(npv, columns=columns, index=rows)


def write_tmat(file_path: str,
               df: pd.DataFrame,
               compress: str = 'zstd',
               compress_header: bool = False,
               dtype: str = None):
    '''
    Write a matrix to a file
    file_path: str
    df: pd.DataFrame
    compress: str, can be zstd, lz4, mmap
    dtype: str, can be float32, float64, int32, int64, int16, int8, None
    '''

    # check if the DataFrame has multiple dtypes
    df_dtypes_ = list(df.dtypes.unique())
    if len(df_dtypes_) > 1:
        raise ValueError('DataFrame has multiple dtypes')
    df_dtype = df_dtypes_[0].name

    if dtype is None:
        dtype = df_dtype

    if dtype != df_dtype:
        df = df.astype(dtype)

    with open(file_path, 'wb') as fout:

        columns = df.columns.to_list()
        rows = df.index.tolist()

        # write magic
        fout.write(b'TMT\n')

        # write params
        params = {
            'header_length': 0,
            'compress': compress,
        }
        fout.write(json.dumps(params).encode('utf-8') + b'\n')

        # write meta data
        js = {
            'columns': columns,
            'rows': rows,
            'dtype': dtype
        }
        js_str = json.dumps(js).encode('utf-8')
        if compress_header:
            js_str = compress_lz4(js_str)
            params['header_length'] = len(js_str)
        else:
            js_str = js_str + b'\n'
        fout.write(js_str)

        data = df.values.tobytes('C')

        if compress:
            if compress == 'mmap':
                pass
            elif compress == 'zstd':
                data = compress_zstd(data)
            elif compress == 'lz4':
                data = compress_lz4(data)
            else:
                raise ValueError(f'Invalid compress method {compress}')

        fout.write(data)

def test_read_write(compress_header=False):
    ofn = '/tmp/test_tmat.tmt'
    df = pd.DataFrame(np.random.rand(100, 100), columns=[
                      f'col{i}' for i in range(100)])
    df.index = [f'row{i}' for i in range(100)]
    write_tmat(ofn, df, dtype='float32', compress='zstd',
               compress_header=compress_header)
    df2 = read_tmat(ofn)
    print(df)
    print(df2)


def test_read_write_mmat(compress_header=False):
    ofn = '/tmp/test_tmat.mmt'
    N = 10000
    M = 10000
    df = pd.DataFrame(np.random.rand(N, M), columns=[
                      f'col{i}' for i in range(M)]).astype('float32')
    df.index = [f'row{i}' for i in range(N)]
    write_tmat(ofn, df, compress='mmap', compress_header=compress_header)
    df2 = read_tmat(ofn)
    print(df)
    print(df2)
    diff = df - df2
    print('max diff', diff.abs().max().max())


def show_tmat(file_path: str):
    df = read_tmat(file_path)
    print(df)


def main():
    from pyttkl import kits

    kits.init_logging('debug')
    args = {'__subcmd__': []}
    args.update(kits.make_sub_cmd(write_tmat))
    args.update(kits.make_sub_cmd(read_tmat))
    args.update(kits.make_sub_cmd(test_read_write))
    args.update(kits.make_sub_cmd(test_read_write_mmat))
    args.update(kits.make_sub_cmd(show_tmat))
    args = kits.make_args(args)
    kits.run_cmds(args)


if __name__ == "__main__":
    main()
