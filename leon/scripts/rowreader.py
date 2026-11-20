# Reads rows of a big .npy by seeking, so the file is not memory-mapped into the process (the OS cache can then be dropped).
import numpy as np


class RowReader:
    def __init__(self, path):
        mm = np.load(path, mmap_mode="r")
        self.shape, self.off = mm.shape, mm.offset
        del mm
        self.row = int(np.prod(self.shape[1:])) * 2
        self.f = open(path, "rb", buffering=0)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        out = np.empty((len(idx),) + tuple(self.shape[1:]), np.uint16)
        for j, i in enumerate(idx):
            self.f.seek(self.off + int(i) * self.row)
            self.f.readinto(memoryview(out[j]).cast("B"))
        return out
