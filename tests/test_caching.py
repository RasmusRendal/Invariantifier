import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from caching import Cache
import unittest


class DummyObject:
    def __init__(self):
        pass


class CacheTester(unittest.TestCase):
    def test_cache_save_get(self):
        try:
            cache = Cache("/tmp/P5/cache-testing")
            cache.save_to_cache("test_cache_save_get", "DummyObject", DummyObject())
            assert cache.exists_in_cache("test_cache_save_get", "DummyObject")

            result = cache.get_from_cache("test_cache_save_get", "DummyObject")
            assert isinstance(result, DummyObject)
        finally:
            cache.remove_from_cache("test_cache_save_get", "DummyObject")

    def test_cache_remove(self):
        cache = Cache("/tmp/P5/cache-testing")
        cache.save_to_cache("test_cache_save_get", "DummyObject", DummyObject())
        assert cache.exists_in_cache("test_cache_save_get", "DummyObject")
        cache.remove_from_cache("test_cache_save_get", "DummyObject")
        assert not cache.exists_in_cache("test_cache_save_get", "DummyObject")


if __name__ == '__main__':
    unittest.main()
