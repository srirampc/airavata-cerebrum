import typing as t
import logging
import itertools
import numpy as np
import numpy.typing as npt

NPIntArray: t.TypeAlias = npt.NDArray[np.integer[t.Any]]

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    barrier = comm.barrier

except ImportError:
    MPI = None
    comm = None
    mpi_rank = 0
    mpi_size = 1
    barrier = lambda: None

def block_low(rank: int, nproc: int, n: int):
    return (rank * n) // nproc 

def block_high(rank: int, nproc: int, n: int):
    return (((rank + 1) * n) // nproc) - 1

def block_size(rank: int, nproc: int, n: int):
    return block_low(rank + 1, nproc, n) - block_low(rank, nproc, n)

def block_owner(j: int, nproc: int, n: int):
    return (((nproc) * ((j) + 1) - 1) // (n))

def block_range(n: int):
    if mpi_size == 1:
        return range(n)
    return range(block_low(mpi_rank, mpi_size, n),
                 block_high(mpi_rank, mpi_size, n) + 1)

def broadcast_props(properties: dict[str, t.Any]) -> dict[str, t.Any]:
    if mpi_size == 1:
        return properties
    return comm.bcast(properties, root=0)

def collect_counts(nc: int):
    if mpi_size == 1:
        return [nc]
    return comm.allgather(nc)

def collect_objects_at_root(robject: object):
    if mpi_size == 1:
        return [robject]
    return comm.gather(robject)

def collect_merge_lists(in_list: list[t.Any]) -> list[t.Any]:
    if mpi_size == 1:
        return in_list
    glist = comm.allgather(in_list)
    return list(itertools.chain.from_iterable(glist))

def collect_merge_lists_at_root(in_list: list[t.Any]) -> list[t.Any]:
    if mpi_size == 1:
        return in_list
    glist = comm.gather(in_list)
    if glist and mpi_rank == 0:
        return list(itertools.chain.from_iterable(glist))
    else:
        return []

def collect_zipmerge_lists_at_root(in_list: list[t.Any]) -> list[t.Any]:
    if mpi_size == 1:
        return in_list
    glist = comm.gather(in_list)
    if glist and mpi_rank == 0:
        return list(
            x
            for x in itertools.chain.from_iterable(
                itertools.zip_longest(*glist)
            )
            if x
        )
    else:
        return []


def accumulate_counts(nc: int):
    if mpi_size == 1:
        return nc
    return comm.allreduce(nc, op=MPI.SUM)

def counts_start_index(ncounts: int):
    if mpi_size == 1:
        return 0
    all_counts = collect_counts(ncounts)
    return 0 if mpi_rank == 0 else sum(all_counts[:mpi_rank])


def gather_np_counts_2d(npnc: NPIntArray):
    if mpi_size == 1:
        return npnc
    rcvbuf = np.empty([mpi_size, npnc.size], dtype=npnc.dtype)
    comm.Allgather(npnc, rcvbuf)
    return rcvbuf

def gather_np_counts_1d(npnc: NPIntArray):
    if mpi_size == 1:
        return npnc
    rcvbuf = np.empty(mpi_size*npnc.size, dtype=npnc.dtype)
    comm.Allgather(npnc, rcvbuf)
    return rcvbuf

def gather_np_counts_1d_at_root(npnc: NPIntArray) -> NPIntArray | None:
    if mpi_size == 1:
        return npnc
    rcvbuf = None
    if mpi_rank == 0:
        rcvbuf = np.empty(mpi_size*npnc.size, dtype=npnc.dtype)
    comm.Gather(npnc, rcvbuf)
    return rcvbuf

def distributed_count_indices(npnc: NPIntArray):
    if mpi_size == 1:
        return np.zeros(npnc.shape, dtype=npnc.dtype), npnc
    rcvbuf = gather_np_counts_2d(npnc)
    return rcvbuf, np.sum(rcvbuf, axis=0)


def log_at_root(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: object,
    **kwargs: t.Any
):
    if mpi_rank == 0:
        logger.log(level, message, *args, **kwargs)

def log_all_comm(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: object,
    **kwargs: t.Any
):
    if logger.isEnabledFor(level):
        barrier()
        prefix = f"PID({mpi_rank}) :: "    
        for pid in range(mpi_size):
            logger.log(level, prefix + message, *args, **kwargs)
            barrier()
