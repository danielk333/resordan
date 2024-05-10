import pickle
from dataclasses import dataclass


def dataset(*args, **kwargs):
    return dataclass(*args, **kwargs, kw_only=True)


@dataset
class BaseDataset:
    def to_pickle(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file_path):
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"Expected un-pickled object to be '{cls}'. Got '{type(obj)}'")
        return obj
