"""
Utilities for testing
"""

import logging
import re
from pathlib import Path
from typing import Dict, TypeVar, Generic, Tuple
from typing import Union
from unittest.mock import MagicMock, call

import numpy as np
from netCDF4 import Variable, Dimension

LOG = logging.getLogger(__name__)

T = TypeVar('T')


class PushableIterator(Generic[T]):
    """
    Wrapper around iterators which ass a peek method to see the next element
    and a method to push elements (back) to the front of it.
    """

    def __init__(self, it):
        self.it = it
        self.returned = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.returned:
            return self.returned.pop()
        else:
            return next(self.it)

    def __push__(self, val):
        self.returned.append(val)

    def __peek__(self):
        if self.returned:
            return self.returned[-1]
        else:
            val = next(self)
            push(self, val)
            return val


def push(it, val):
    it.__push__(val)


def peek(it, val):
    it.__peek__(val)


def mock_dataset_meta(*header_path: Tuple[Union[Path, str]]):
    if not isinstance(header_path, Path):
        header_path = Path(*header_path)

    ds = MagicMock()
    dims = {}  # type: Dict[str, Union[Dimension MagicMock]]
    vars = {}  # type: Dict[str, Union[Variable, MagicMock]]

    DATATYPE_MAP = dict(
        float='float32',
        int='int32',
        char='|S1'
    )

    RE_HEADER = re.compile('netcdf (\S+) \{')
    RE_HEADER_DIMENSIONS = re.compile('^dimensions:$')
    RE_DIMENSION = re.compile('^\t(\w+) = (\w+) *;')
    RE_HEADER_VARIABLES = re.compile('^variables:$')
    RE_VARIABLE = re.compile('^\t(\w+) (\w+)\(([^)]*)\) *;$')
    RE_HEADER_GLOBALS = re.compile('^\W*global attributes:$')
    RE_GLOBAL_ATTRIB = re.compile('\t\t:([\w-]+) *= * (.+?) *;$')

    RE_QUOTED = re.compile('^"(.*?)"')
    RE_COMMA_SEPARATOR = re.compile('\s*,\s*')

    def parse_value(v: str):
        m = RE_QUOTED.match(v)
        if m:
            return m.group(1)

        errors = []
        try:
            return np.int32(v)
        except ValueError as e:
            errors.append(e)

        if v.endswith('f'):
            v = v[:-1]
        try:
            return np.float32(v)
        except ValueError as e:
            errors.append(e)

        a = RE_COMMA_SEPARATOR.split(v)
        if len(a) > 1:
            return [parse_value(x) for x in a]

        raise ValueError('Unable to parse %s: %s', v, errors)

    def add_dim(name: str, size: str):
        dims[name] = dim = MagicMock(name=name)
        dim.name = name
        dim.size = None if size == 'UNLIMITED' else int(size)
        dim.isunlimited = (lambda self: lambda: self.size is None)(dim)  # binding self to dim
        dim.__repr__ = lambda self: 'MockDim: {} = {}'.format(dim.name, dim.size)
        LOG.debug('%s', dim)
        return dim

    def read_dims(lines: PushableIterator[str]):
        for line in lines:
            m = RE_DIMENSION.match(line)
            if not m:
                return push(lines, line)
            add_dim(*m.groups())

    def add_var_attribs(var, lines: PushableIterator[str]):
        RE_VARIABLE_ATTRIB = re.compile('^\t\t{}: *(\w+) *= *(.+?) *;$'.format(var.name))
        for line in lines:
            m = RE_VARIABLE_ATTRIB.match(line)
            if not m:
                # LOG.info('Not an attribute of %s: %s', name, line)
                return push(lines, line)
            att_name, att_value = m.groups()
            att_value = parse_value(att_value)
            setattr(var, att_name, att_value)
            LOG.debug('     %s = %s', att_name, att_value)

    def add_var(lines: PushableIterator[str], datatype: str, name: str, dimensions: str):
        var = vars[name] = MagicMock(name=name)
        var.name = name
        var.datatype = np.dtype(DATATYPE_MAP.get(datatype, datatype))
        var.dimensions = tuple(dimensions.split(', '))
        var.__repr__ = lambda self: 'MockVar: {} {}{}'.format(self.datatype, self.name, self.dimensions)
        LOG.debug('%s', var)
        add_var_attribs(var, lines)
        return var

    def read_vars(lines: PushableIterator[str]):
        for line in lines:
            m = RE_VARIABLE.match(line)
            if not m:
                return push(lines, line)
            add_var(lines, *m.groups())

    def read_global_attribs(lines: PushableIterator[str]):
        for line in lines:
            m = RE_GLOBAL_ATTRIB.match(line)
            if not m:
                # LOG.debug('Not a global attribute: %s', line)
                return push(lines, line)
            att_name, att_value = m.groups()
            att_value = parse_value(att_value)
            setattr(ds, att_name, att_value)

    with header_path.open() as lines:
        lines = (l.rstrip() for l in lines)
        lines = PushableIterator(lines)
        line = next(lines)
        wrfout_name = RE_HEADER.match(line).group(1)

        ds = MagicMock(name=wrfout_name)
        ds.dimensions = dims
        ds.variables = vars

        LOG.debug('Mocking wrfout file: %s', wrfout_name)
        for line in lines:
            if RE_HEADER_DIMENSIONS.match(line):
                read_dims(lines)
            elif RE_HEADER_VARIABLES.match(line):
                read_vars(lines)
            elif RE_HEADER_GLOBALS.match(line):
                read_global_attribs(lines)
            else:
                # LOG.info('ignored:"%s"', line)
                pass

    return ds


def nda_to_code(name: str, nda: np.ndarray, digits: int = 0):
    """
    Writes a numpy array as a compact call to nda_from_string which can be pasted into code to recreate it.

    :param name: Name of variable to create
    :param nda: numpy array to write out
    :param digits: Number of digits to keep
    """
    shape = nda.shape
    nda = nda.ravel().tolist()
    mul = 10 ** (-digits)
    nda = [str(int(x / mul + 0.5)) for x in nda]
    s = ' '.join(nda)
    print('{} = nda_from_string({}, {}, \'{}\')'.format(name, shape, digits, s))


def nda_from_string(shape, digits, s):
    a = s.split(' ')
    a = np.array(a, dtype=np.float)
    a = a.reshape(*shape)
    a = a * 10 ** (-digits)
    # print(len(s), '->', a.shape)
    return a


class _SLICE_CALL(object):
    def __getitem__(self, slices):
        return call(slices)


slice_call = _SLICE_CALL()
