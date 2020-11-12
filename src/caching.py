"""caching module"""
import pickle
import os


class Cache:
    """class for unified caching of objects in pickled form"""
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, mode=0o755)

    def get_from_cache(self, namespace, identifier):
        """get an object from the cache given a namespace and identifier"""
        with open(f"{self.cache_dir}{namespace}_{identifier}.pickle", "rb") as file_pointer:
            pickled_object = pickle.load(file_pointer)
        return pickled_object

    def save_to_cache(self, namespace, identifier, python_object):
        """save an object to the cache given a namespace and identifier"""
        with open(f"{self.cache_dir}{namespace}_{identifier}.pickle", "wb") as file_pointer:
            pickle.dump(python_object, file_pointer)

    def remove_from_cache(self, namespace, identifier):
        """remove an object from the cache given a namespace and identifier"""
        filename = f"{self.cache_dir}{namespace}_{identifier}.pickle"
        if os.path.exists(filename):
            os.remove(filename)

    def exists_in_cache(self, namespace, identifier):
        """check whether a given namespace-identifier combo exists in the cache"""
        return os.path.exists(f"{self.cache_dir}{namespace}_{identifier}.pickle")
