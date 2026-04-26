# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import gc
import numpy as np
import tempfile
from pathlib import Path


cdef class Memory_Manager:
    cdef public object temp_dir
    cdef list _files
    cdef list _arrays

    def __init__(self, base_dir=None):
        if base_dir is not None:
            self.temp_dir = Path(base_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='entrap_'))
        self._files = []
        self._arrays = []

    def create(self, tuple shape, dtype=np.float64, name=None):
        cdef str fname_str
        fname = self.temp_dir / (
            f'{name}.dat' if name else f'memmap_{len(self._files)}.dat'
        )
        fname_str = str(fname)
        self._files.append(fname)
        mmap = np.memmap(fname_str, dtype=dtype, mode='w+', shape=shape)
        self._arrays.append(mmap)
        return mmap

    def cleanup(self):
        for arr in self._arrays:
            try:
                arr.flush()
                del arr
            except Exception:
                pass
        self._arrays.clear()
        gc.collect()
        for fname in self._files:
            try:
                if fname.exists():
                    fname.unlink()
            except Exception:
                pass
        self._files.clear()
        try:
            if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                self.temp_dir.rmdir()
        except Exception:
            pass

    def __dealloc__(self):
        self.cleanup()
