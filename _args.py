# %%
import argparse
import inspect
import logging

logger = logging.getLogger(__name__)

def make_args(args, root_parser=None):
    ''' e.g.
        def test_1():
            test_str_1 = '20210916 --DROOT ~/DROOT'.split()
            args = make_args([
                ['date', 'date'],
                ['--DROOT', 'DROOT', None, str]
            ]).parse_args(test_str_1)
            print (vars(args))
        def test_2():
            def f0(args):
                print('f0: ', args)
            test_str = 'next_tds 20210916 --DROOT ~/DROOT'.split()
            args = make_args({
                '__subcmd__': True,
                'next_tds': {
                    'func': f0,
                    'args':
                        [['date', 'date'],
                        ['--DROOT', 'DROOT', None, str]]
                }
            }).parse_args(test_str)
            args.func(args)
            print (vars(args))
    '''
    root_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    def make_parser(parser, args):
        for segs in args:
            if not segs[0].startswith('--'):
                dtype = str
                if len(segs) > 2:
                    name, hstr, dtype = segs
                elif len(segs) > 1:
                    name, hstr = segs
                else:
                    name = segs[0]
                    hstr = name
                parser.add_argument(name, type=dtype, help=hstr)
            else:
                name, hstr, default, dtype = segs
                if dtype == bool:
                    parser.add_argument(
                        name, default=default, help=hstr, action='store_true')
                else:
                    parser.add_argument(
                        name, default=default, type=dtype, help=hstr)
    if isinstance(args, list):
        make_parser(root_parser, args)
    elif isinstance(args, dict):
        subcmd = args.get('__subcmd__')
        if subcmd is not None:
            make_parser(root_parser, args['__subcmd__'])
            keys = set(args.keys()) - set(['__subcmd__'])
            subps = root_parser.add_subparsers(help='sub-command help')
            for key in keys:
                subp = subps.add_parser(key, help=args[key].get('help', None),
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                make_parser(subp, args[key]['args'])
                subp.set_defaults(func=args[key]['func'])
    return root_parser

# %%


def test():
    def test_1():
        test_str_1 = '20210916 --DROOT ~/DROOT'.split()
        args = make_args([
            ['date', 'date'],
            ['--DROOT', 'DROOT', None, str]
        ]).parse_args(test_str_1)
        print(vars(args))

    def test_2():
        def f0(args):
            print('f0: ', args)
        test_str = 'next_tds 20210916 --DROOT ~/DROOT'.split()
        args = make_args({
            '__subcmd__': True,
            'next_tds': {
                'func': f0,
                'args':
                    [['date', 'date'],
                     ['--DROOT', 'DROOT', None, str]]
            }
        }).parse_args(test_str)
        args.func(args)
        print(vars(args))

    def trash_1():
        from etkits import kits
        args = kits.make_args([['date', '']]).parse_args()
    args = test_2()


def run_cmds(argp: argparse.ArgumentParser, arg_str=None):
    args = argp.parse_args(arg_str)
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt as e:
            logger.warn('KeyboardInterrupt')
        logger.info('exit')
    else:
        argp.print_help()


def remove_func(args):
    d = vars(args)
    del d['func']
    return d


def GETARGS(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def make_sub_cmd(func, name=None, helps={}):
    if name is None:
        name = func.__name__
    rets = {
        'func': lambda args: func(**remove_func(args)),
    }
    signature = inspect.signature(func)
    args = []
    for k, v in signature.parameters.items():
        if v.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if v.default is None:
            continue
        if v.default is inspect.Parameter.empty:
            args.append([k, helps.get(k, k)])
        else:
            args.append([f"--{k}", helps.get(k, k), v.default, type(v.default)])
    rets.update({'args': args})
    return {name: rets}


if __name__ == "__main__":
    test()
