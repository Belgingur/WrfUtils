from nose.tools import assert_equal
from utilities import make_regexp


def test_make_regexp():
    assert_equal('^lv\.is\.200$', make_regexp('lv.is.200'))
    assert_equal('^lv\..*\.200$', make_regexp('lv.%.200'))
    assert_equal('^lv\.fo\.[1-4][0-9]$', make_regexp('lv.fo.[1-4][0-9]'))
    assert_equal('^lv\..*\..*$', make_regexp('lv.%.%'))
