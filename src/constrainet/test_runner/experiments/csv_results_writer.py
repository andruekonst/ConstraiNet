import pandas as pd
from pathlib import Path
import logging
from .base import AbstractResultsWriter


class CSVResultsWriter(AbstractResultsWriter):
    def __init__(self, path: Path, enable_temporary: bool = False):
        self.path = path
        if enable_temporary:
            self.tmp_path = path.parent / (path.stem + '__temp.csv')
        else:
            self.tmp_path = None
        self._collection = dict()
        self.count = 0

    def open(self):
        assert not self.path.exists(), f'Results path {self.path!r} should not exist before open'

    def close(self):
        logging.info(f'Saving results into {self.path!r}')
        self.flush(self.path)

    def append(self, params: dict):
        for key in self._collection.keys():
            if key not in params:
                self._collection[key].append(None)

        for k, v in params.items():
            if k not in self._collection:
                self._collection[k] = [None] * self.count
            self._collection[k].append(v)

        self.count += 1
        if self.tmp_path is not None:
            self.flush(self.tmp_path)

    def flush(self, path: Path):
        pd.DataFrame(self._collection).to_csv(path)
