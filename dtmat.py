from .tmat import compress_lz4, compress_zstd, decompress_lz4, decompress_zstd
from .tmat import str_to_dtype, logging_trace
import json
import numpy as np
import pandas as pd
import mmap
import logging
from pyttkits import kits
import pandas as pd
import os
import errno
import fcntl
import time
from datetime import datetime
from collections import defaultdict
import os
import tqdm
import numpy as np
import functools

import logging
import duckdb

logger = logging.getLogger(__name__)


def to_upper_power2(size):
    '''
    Get the next power of 2
    '''
    if size == 0:
        return 1
    size -= 1
    size |= size >> 1
    size |= size >> 2
    size |= size >> 4
    size |= size >> 8
    size |= size >> 16
    size |= size >> 32
    return size + 1


def write_dtmat(file_path: str,
                df: pd.DataFrame,
                compress: str = 'zstd',
                compress_header: bool = False,
                dtype: str | None = None,
                remove_unused_levels: bool = False):
    '''
    Write a matrix to a file
    file_path: str
    df: pd.DataFrame
    compress: str, can be zstd, lz4, mmap
    dtype: str, can be float32, float64, int32, int64, int16, int8, None
    '''

    if remove_unused_levels:
        df.index = df.index.remove_unused_levels()


    df_dtypes = df.dtypes
    if isinstance(df_dtypes, pd.Series):
        df_dtypes_ = list(df.dtypes.unique())
        if len(df_dtypes_) > 1:
            raise ValueError('DataFrame has multiple dtypes')
        df_dtype = df_dtypes_[0].name
    else:
        df_dtype = df_dtypes.name

    if dtype is None:
        dtype = df_dtype

    if dtype != df_dtype:
        df = df.astype(dtype)

    columns = df.columns.to_list()
    levels = [list(x) for x in df.index.levels]

    logger.debug("write dtmat: %s", file_path)
    with open(file_path, 'wb') as fout:

        # write magic
        fout.write(b'TMT\n')

        # write params

        # write meta data
        js = {
            'columns': columns,
            'level_names': df.index.names,
            'levels': levels,
        }
        js_str = json.dumps(js).encode('utf-8')
        if compress_header:
            js_str = compress_lz4(js_str)
        else:
            js_str = js_str + b'\n'

        header_length = len(js_str)
        header_capacity = max(to_upper_power2(header_length) * 2, 16 * 1024)

        params = {
            'header_length': header_length,
            'header_capacity': header_capacity,
            'compress': compress,
            'dtype': dtype,
            'total_offset': 4 + 1024 + header_capacity,
        }
        params_str = json.dumps(params).encode('utf-8') + b'\n'
        params_str_capacity = 1024

        fout.write(params_str)
        fout.write(b'\0' * (params_str_capacity - len(params_str)))
        fout.write(js_str)
        fout.write(b'\0' * (header_capacity - len(js_str)))

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


class DTMat:

    def __init__(self, file_path: str, read_only: bool = True):
        self.file_path = file_path
        self._df = None
        self._params = None
        self._data = None
        self._read_only = read_only
        self._compress = None

        self.fn = None

    @staticmethod
    def open(file_path: str, read_only: bool = True):
        '''
        Read a matrix from a file
        file_path: str
        read_only: bool, if True, the file is opened in read-only mode
        '''
        dtmat = DTMat(file_path, read_only)
        dtmat._read()
        return dtmat


    def _read(self):
        fin = self.fn = open(self.file_path, 'r+b' if not self._read_only else 'rb')
            # read magic
        magic = fin.read(4)
        if magic != b'TMT\n':
            raise ValueError('Invalid file format')

        # read params
        params_str = fin.read(1024).strip(b'\0')
        params = json.loads(params_str.decode('utf-8'))
        self._params = params
        # read meta data
        js_str = fin.read(params['header_length'])
        if self._params['compress'] == 'lz4':
            js_str = decompress_lz4(js_str)
        # logger.debug("js_str: %s", js_str)
        js = json.loads(js_str.decode('utf-8'))

        self._meta = js
        comppress_mode = self._params['compress']
        total_offset = self._params['total_offset']
        self._compress = comppress_mode

        self._load()


    def _load(self):
        comppress_mode = self._params['compress']
        total_offset = self._params['total_offset']
        js = self._meta
        if comppress_mode == 'mmap':
            # mmap remaining data
            self._data = np.memmap(self.file_path,
                                   dtype=self._params['dtype'],
                                   mode='r',
                                   offset=total_offset)

            # logger.debug("mmap data shape: %s", self._data.shape)
        else:
            self._data = self.fn.read()
            if comppress_mode == 'zstd':
                self._data = decompress_zstd(self._data)
            elif comppress_mode == 'lz4':
                self._data = decompress_lz4(self._data)
            else:
                raise ValueError(f'Invalid compress method {comppress_mode}')

        npv = np.frombuffer(self._data, dtype=self._params['dtype']).reshape(-1, len(js['columns']))

        index = pd.MultiIndex.from_product(js['levels'],
                                           names=js['level_names'])

        # logger.debug("npv shape: %s", npv.shape)

        # logger.debug("index shape: %s", index)

        df = pd.DataFrame(npv, columns=js['columns'], index=index)
        self._df = df


    def __del__(self):
        if self.fn is not None:
            self.fn.close()


    def df(self):
        '''
        Return the DataFrame
        '''
        if self._df is None:
            self._read()
        return self._df

    def _write_header(self):
        header_js = self._meta
        header_js_str = json.dumps(header_js).encode('utf-8') + b'\n'

        # logger.debug("header_js_str: %s", header_js_str)

        self._params['header_length'] = len(header_js_str)

        # logger.debug("new_params: %s", self._params)

        if self._params['header_length'] > self._params['header_capacity']:
            raise ValueError('Header length is greater than header capacity')
        self.fn.seek(0, 0)
        self.fn.write(b'TMT\n')
        params_str = json.dumps(self._params).encode('utf-8') + b'\n'
        params_str_capacity = 1024
        self.fn.write(params_str)
        self.fn.seek(4 + params_str_capacity, 0)
        if self._compress == 'lz4':
            header_js_str = compress_lz4(header_js_str)
        self.fn.write(header_js_str)
        self.fn.flush()

    def __repr__(self):
        return repr(self._df)

    def __str__(self):
        return str(self._df)

    def append_df(self, df: pd.DataFrame):
        '''
        Append a DataFrame to the matrix
        df: pd.DataFrame
        '''
        if df.index.nlevels != self._df.index.nlevels:
            raise ValueError('DataFrame has different number of levels')

        if any(df.columns != self._df.columns):
            raise ValueError('DataFrame has different columns')

        if list(df.dtypes) != list(self._df.dtypes):
            raise ValueError('DataFrame has different dtypes')

        if len(df.index.levels[1]) != len(self._df.index.levels[1]):
            raise ValueError('DataFrame has different levels')

        self._meta['levels'][0] += list(df.index.levels[0])

        if self._compress == 'mmap':
            data = df.values.tobytes('C')
            # logger.debug("data shape: %s", len(data))
            self.fn.seek(0, 2)
            self.fn.write(data)
            self.fn.flush()
            self._write_header()
            self.fn.close()
            self.fn = open(self.file_path, 'r+b')

            self._load()
        else:
            raise ValueError(f'not support append for {self._compress}')

class DTMatDB:

    def __init__(self, dbpath='/data/public/futures/tscache/bar/sandbox/tian/ctadmatv0/'):
        self.dbpath = os.path.expanduser(dbpath)
        self.interval = '1m'  # Default interval

    def use_interval(self, interval):
        """
        Set the interval for the database operations.
        """
        self.interval = interval
        return self

    def _path(self, symbol, name, interval=''):
        if interval == '':
            interval = self.interval
        ofn = f'{self.dbpath}/{interval}/{name}/{symbol}.dtmat'
        return ofn

    def get(self, name, symbol, dates='', interval=''):
        if interval == '':
            interval = self.interval
        # ofn = '/data/public/futures/tscache/bar/sandbox/tian/ctadmatv0/1m/perp/klines/FwdVWAP_4h/SS_BTCETH.dtmat'
        ofn = self._path(symbol, name, interval)
        if not os.path.exists(ofn):
            raise ValueError(f"file not found: {ofn}")

        df = DTMat.open(ofn).df()

        if dates != '':
            start_date, end_date = dates.split('-')
            df = df.loc[start_date:end_date]

        return df

    def write(self, name, symbol, df, interval='', compress='mmap', remove_unused_levels=False):
        if interval == '':
            interval = self.interval
        ofn = self._path(symbol, name, interval)
        if not os.path.exists(os.path.dirname(ofn)):
            os.makedirs(os.path.dirname(ofn), exist_ok=True)

        from pyttkl.dtmat import write_dtmat

        write_dtmat(ofn, df, compress=compress, remove_unused_levels=remove_unused_levels)

        return ofn

    def has(self, name, symbol, dates_range='', interval=''):
        if interval == '':
            interval = self.interval
        ofn = self._path(symbol, name, interval)
        if dates_range != '':
            df = self.get(symbol, name, dates_range, interval)
            start_date, end_date = dates.split('-')
            if start_date <= df.index[0] and end_date >= df.index[-1]:
                return True
        else:
            return os.path.exists(ofn)


def test_read_dtmat(ifn):
    '''
    Test read dtmat
    '''
    dtmat = DTMat.read(ifn)
    print(dtmat.df())


def test_append_dtmat(ofn):
    index_ = pd.MultiIndex.from_product([list(range(3)), list(range(2))])
    print(index_)
    print(index_.levels)
    df = pd.DataFrame(np.arange(12).reshape(6, 2), index=index_)
    df.columns = ['a', 'b']
    df.index.names = ['DATE', 'TIME']
    df = df.astype('int32')
    write_dtmat(ofn, df, compress='mmap', compress_header=False)

    dtmat = DTMat.open(ofn, read_only=False)
    print(dtmat.df())

    index_ = pd.MultiIndex.from_product([list(range(3, 1024)), list(range(2))])
    df2 = pd.DataFrame(np.arange(0, 2 * (1024 - 3) * 2).reshape(-1, 2), index=index_)
    df2.columns = ['a', 'b']
    df2.index.names = ['DATE', 'TIME']
    df2 = df2.astype('int32')

    dtmat.append_df(df2)

    print('new_df after append')
    print(dtmat.df())


def test_write_dtmat_db():
    db = DTMatDB('/data/public/futures/tscache/bar/sandbox/tian/ctadmatv_test/')
    index_ = pd.MultiIndex.from_product([list(range(3)), list(range(2))])
    df = pd.DataFrame(np.arange(12).reshape(6, 2), index=index_)
    df.columns = ['a', 'b']
    df.index.names = ['DATE', 'TIME']
    df = df.astype('int32')

    df = df.loc[[1]]
    print(df)

    db.write('perp/klines/FwdVWAP_4h', 'SS_BTCETH', df, compress='mmap', remove_unused_levels=True)

def main():
    from pyttkl import kits

    kits.init_logging('debug')
    args = {'__subcmd__': []}
    args.update(kits.make_sub_cmd(test_read_dtmat))
    args.update(kits.make_sub_cmd(test_append_dtmat))
    args.update(kits.make_sub_cmd(test_write_dtmat_db))
    args = kits.make_args(args)
    kits.run_cmds(args)


if __name__ == "__main__":
    main()
