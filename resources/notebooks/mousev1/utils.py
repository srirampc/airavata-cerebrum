import h5py
import numpy as np

from enum import Enum
class MVParMethod(str, Enum):
    NONE = 'NONE'
    ALL_GATHER = 'ALL_GATHER'
    ALL_GATHER_BY_SND_RCV = 'ALL_GATHER_BY_SND_RCV'
    DISTRIBUTED = 'DISTRIBUTED'


def add_hdf5_attrs(hdf5_handle: h5py.File):
    # TODO: move this as a utility function
    hdf5_handle['/'].attrs['magic'] = np.uint32(0x0A7A)
    hdf5_handle['/'].attrs['version'] = [np.uint32(0), np.uint32(1)]


def block_low(rank: int, nproc: int, n: int):
    return (rank * n) // nproc

def block_high(rank: int, nproc: int, n: int):
    return (((rank + 1) * n) // nproc) - 1

def block_size(rank: int, nproc: int, n: int):
    return block_low(rank + 1, nproc, n) - block_low(rank, nproc, n)

def block_owner(j: int, nproc: int, n: int):
    return (((nproc) * ((j) + 1) - 1) // (n))



