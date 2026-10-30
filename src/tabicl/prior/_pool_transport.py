"""Node-local data plane, independent of sampling and scheduling.

The service owns a private directory and unlinks each table after the client
maps it. Existing mappings stay alive through NumPy/Torch ownership, including
after service shutdown. No multiprocessing resource_tracker registrations or
Tensor file-descriptor RPCs are required. Prefer tmpfs; never use OSS storage.
"""
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


class OutputStore:
    def __init__(self, transport):
        self.transport = transport
        root = "/dev/shm" if os.path.isdir("/dev/shm") else None
        self.directory = tempfile.TemporaryDirectory(prefix="tabicl-output-", dir=root) if transport == "mmap" else None

    @property
    def path(self):
        return self.directory.name if self.directory else None

    def release(self, results):
        if self.directory:
            for descriptor, _, _ in results:
                path = Path(descriptor[0])
                if path.parent != Path(self.path):
                    raise ValueError("output outside owned directory")
                path.unlink(missing_ok=True)

    def close(self):
        if self.directory:
            self.directory.cleanup()


def encode_table(x, y, directory=None):
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
    if directory is None:
        return x, y
    offset = (x.nbytes + 63) // 64 * 64
    fd, path = tempfile.mkstemp(dir=directory, prefix="table-")
    try:
        os.ftruncate(fd, offset + y.nbytes)
        # Fail with ENOSPC instead of letting a later mmap write SIGBUS a worker.
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(fd, 0, offset + y.nbytes)
        output = np.memmap(path, mode="r+", dtype=np.uint8, shape=(offset + y.nbytes,))
        np.ndarray(x.shape, x.dtype, buffer=output)[:] = x
        np.ndarray(y.shape, y.dtype, buffer=output, offset=offset)[:] = y
        del output
        return (path, x.shape, x.dtype.str, y.shape, y.dtype.str, offset), None
    except BaseException:
        os.unlink(path)
        raise
    finally:
        os.close(fd)


def decode_batch(payload, transport):
    xs, ys, ds, lengths, trains = payload
    if transport == "mmap":
        xx, yy = [], []
        for path, xshape, xdtype, yshape, ydtype, offset in xs:
            # Copy-on-write mapping is writable for Torch, but client changes
            # cannot affect the shared file or another client's view.
            backing = np.memmap(path, mode="c", dtype=np.uint8)
            xx.append(torch.from_numpy(np.ndarray(xshape, xdtype, buffer=backing)))
            yy.append(torch.from_numpy(np.ndarray(yshape, ydtype, buffer=backing, offset=offset)))
        xs, ys = xx, yy
    else:
        xs, ys = [torch.from_numpy(x) for x in xs], [torch.from_numpy(y) for y in ys]
    return xs, ys, torch.from_numpy(ds), torch.from_numpy(lengths), torch.from_numpy(trains)
