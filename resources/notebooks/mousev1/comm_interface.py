import typing as t
import logging
import itertools
import numpy as np
import numpy.typing as npt
import pickle
import pandas as pd
import hashlib

from typing_extensions import override
from airavata_cerebrum.util.profile import (
    ProfilesT,
    timing_profile,
    memory_profile,
    timestamp_prefix,
    log_mem_usage,
    log_data_frame,
    log_with_timestamp
)

NPIntArray: t.TypeAlias = npt.NDArray[np.integer[t.Any]]

LOGGER = logging.getLogger(__name__)
MAX_XFER_LIMIT = (1024*1024*1024) + (512*1024*1024) # 1.5 GB for now
MPI_fail_params_nonuniform = True  # raise exception if params across MPI procs are not same.


def block_low(rank: int, nproc: int, n: int):
    return (rank * n) // nproc

def block_high(rank: int, nproc: int, n: int):
    return (((rank + 1) * n) // nproc) - 1

def block_size(rank: int, nproc: int, n: int):
    return block_low(rank + 1, nproc, n) - block_low(rank, nproc, n)

def block_owner(j: int, nproc: int, n: int):
    return (((nproc) * ((j) + 1) - 1) // (n))


class CommInterface:
    def __init__(self):
        self._rank: int = 0
        self._size: int = 1

    @property
    def rank(self):
        return self._rank

    @property
    def size(self):
        return self._size

    def barrier(self):
        pass

    def block_range(self, n: int):
        return range(n)

    def broadcast(self, value: t.Any, _source: int = 0) -> t.Any:
        return value

    def collect_counts(self, nc: int) -> list[int]:
        return [nc]

    def collect_objects_at_root(self, robject: object) -> list[object] | None:
        return [robject]

    def collect_merge_lists(self, in_list: list[t.Any]) -> list[t.Any] | None:
        return in_list

    def accumulate_counts(self, nc: int) -> int:
        return nc

    def counts_start_index(self, _ncounts: int) -> int:
        return 0

    def gather_np_counts_2d(self, npnc: NPIntArray) -> NPIntArray:
        return npnc

    def gather_np_counts_1d(self, npnc: NPIntArray) -> NPIntArray:
        return npnc

    def gather_np_counts_1d_at_root(self, npnc: NPIntArray) -> NPIntArray | None:
        return npnc

    def distributed_count_indices(
        self,
        npnc: NPIntArray
    ) -> tuple[NPIntArray, NPIntArray]:
        return np.zeros(npnc.shape, dtype=npnc.dtype), npnc

    def collect_merge_lists_at_root(
        self,
        in_list: list[t.Any],
        _use_snd_rcv: bool = False,
    ) -> list[t.Any]:
        return in_list

    def collect_zipmerge_lists_at_root(
        self,
        in_list: list[t.Any],
        _use_snd_rcv: bool = False,
    ) -> list[t.Any]:
        return in_list

    def log_at_root(
        self,
        logger: logging.Logger,
        level: int,
        message: str,
        *args: object,
        **kwargs: t.Any
    ):
        if logger.isEnabledFor(level) and self.rank == 0:
            log_with_timestamp(logger, level, message, *args,
                               rank=self.rank, **kwargs)

    def log_comm(
        self,
        logger: logging.Logger,
        level: int,
        message: str,
        *args: object,
        **kwargs: t.Any
    ):
        if not logger.isEnabledFor(level):
            return
        log_msg = f"{timestamp_prefix(self.rank)} {message.format(args, kwargs)}"
        log_messages = self.collect_objects_at_root(log_msg)
        if self.rank == 0 and log_messages:
            for msg in log_messages:
                logger.log(level, msg)

    def summarize_profiles(self) -> tuple[ProfilesT, ProfilesT]:
        run_times_rs = self.collect_merge_lists_at_root(timing_profile(self.rank))
        mem_used_rs = self.collect_merge_lists_at_root([memory_profile(self.rank)])
        if run_times_rs and mem_used_rs and self.rank == 0:
            return run_times_rs, mem_used_rs
        else:
            return [{}], [{}]

    def log_profile_summary(
        self,
        logger: logging.Logger,
        level: int,
        times_out_file: str | None = None,
        mem_out_file: str | None = None,
    ):
        if not logger.isEnabledFor(level):
            return
        timers_summary, mem_summary = self.summarize_profiles()
        if self.rank == 0:
            timers_df = pd.DataFrame(data=timers_summary)
            mem_df = pd.DataFrame(data=mem_summary)
            # print the dataframe
            log_data_frame(logger, level, timers_df)
            log_data_frame(logger, level, mem_df)
            if times_out_file:
                timers_df.to_csv(times_out_file)
            if mem_out_file:
                mem_df.to_csv(mem_out_file)

    def log_ps_profile(
        self,
        logger: logging.Logger,
        level: int,
    ):
        if not logger.isEnabledFor(level):
            return
        dxrss = memory_profile(self.rank)
        rslst = self.collect_objects_at_root(dxrss)
        if self.rank == 0:
            mem_df = pd.DataFrame(data=rslst)
            log_data_frame(logger, level, mem_df)
            log_mem_usage(logger, level)

    def check_properties_across_ranks(
        self,
        _properties: dict[str, t.Any],
        _graph_type: str='node'
    ) -> None:
        return None


class IDComm(CommInterface):
    def __init__(self,):
        super().__init__()
        self._rank: int = 0
        self._size: int = 1

_default_comm_instance : CommInterface  = IDComm()
_comm_type : type[CommInterface]  = IDComm

try:
    from mpi4py import MPI

    class MPIComm(CommInterface):
        def __init__(
            self,
            comm: MPI.Comm  = MPI.COMM_WORLD
        ):
            super().__init__()
            self._comm: MPI.Comm = comm
            self._rank: int = comm.Get_rank()
            self._size: int = comm.Get_size()

        @override
        def barrier(self):
            self._comm.barrier()

        @override
        def block_range(self, n: int):
            return range(block_low(self.rank, self.size, n),
                         block_high(self.rank, self.size, n) + 1)

        # wrappers around mpi collective communication routines
        @override
        def broadcast(self, value: t.Any, source: int = 0) -> t.Any:
            return self._comm.bcast(value, root=source)

        @override
        def collect_counts(self, nc: int) -> list[int]:
            return self._comm.allgather(nc)

        @override
        def collect_objects_at_root(self, robject: object) -> list[object] | None:
            return self._comm.gather(robject)

        @override
        def collect_merge_lists(self, in_list: list[t.Any]) -> list[t.Any] | None:
            glist = self._comm.allgather(in_list)
            return list(itertools.chain.from_iterable(glist))

        def send_bytes_in_batches(self, sx: bytes, dest: int):
            nblen = len(sx)
            # Send the size of data
            self._comm.send(nblen, dest=dest, tag=11)
            starts = list(range(0, nblen, MAX_XFER_LIMIT))
            #
            # This recv call is to make sure I wait until the
            # destination processor is ready to accept from me
            rstarts = self._comm.recv(source=dest, tag=12)
            assert starts == rstarts , f"Starts {starts} != {rstarts} @ {self.rank}"
            ends = starts[1:] + [nblen]
            # Send by batches
            for s,e in zip(starts, ends):
                self._comm.send(sx[s:e], dest=dest, tag=13)

        def recieve_bytes_in_batches(self, src: int):
            nblen: int = self._comm.recv(source=src, tag=11)
            # Send the size of data
            rbytes = bytearray(nblen)
            starts = list(range(0, nblen, MAX_XFER_LIMIT))
            #
            # Next send call is to make sure the sending processor
            # waits until I am ready to accept from me
            self._comm.send(starts, dest=src, tag=12)
            ends = starts[1:] + [nblen]
            # print(starts, ends)
            # Send by batches
            for s,e in zip(starts, ends):
                rbytes[s:e] = self._comm.recv(source=src, tag=13)
            return rbytes

        def send_object_in_batches(self, in_object: object, dest: int):
            sx = pickle.dumps(in_object, protocol=pickle.HIGHEST_PROTOCOL)
            self.send_bytes_in_batches(sx, dest)

        def recieve_object_in_batches(self, src: int):
            sx = self.recieve_bytes_in_batches(src)
            return pickle.loads(sx)

        @override
        def accumulate_counts(self, nc: int):
            return self._comm.allreduce(nc, op=MPI.SUM)

        @override
        def counts_start_index(self, ncounts: int):
            all_counts = self.collect_counts(ncounts)
            return 0 if self.rank == 0 else sum(all_counts[:self.rank])

        @override
        def gather_np_counts_2d(self, npnc: NPIntArray):
            rcvbuf = np.empty([self.size, npnc.size], dtype=npnc.dtype)
            self._comm.Allgather(npnc, rcvbuf)
            return rcvbuf

        @override
        def gather_np_counts_1d(self, npnc: NPIntArray):
            rcvbuf = np.empty(self.size*npnc.size, dtype=npnc.dtype)
            self._comm.Allgather(npnc, rcvbuf)
            return rcvbuf

        @override
        def gather_np_counts_1d_at_root(self, npnc: NPIntArray) -> NPIntArray | None:
            rcvbuf = None
            if self.rank == 0:
                rcvbuf = np.empty(self.size*npnc.size, dtype=npnc.dtype)
            self._comm.Gather(npnc, rcvbuf)
            return rcvbuf

        @override
        def distributed_count_indices(self, npnc: NPIntArray):
            rcvbuf = self.gather_np_counts_2d(npnc)
            return rcvbuf, np.sum(rcvbuf, axis=0)


        def gather_at_root_by_snd_rcv(self, in_list: list[t.Any]) -> list[t.Any]:
            glist = []
            #
            # dump_pickle(f"tmp/in_list_{mpi_rank}.pickle", in_list)
            # barrier()
            for i in range(self.size):
                if i == 0 and self.rank == 0:
                    glist.append(in_list)
                    continue
                if self.rank == i:
                    try:
                        self.send_object_in_batches(in_list, dest=0)
                    except MPI.Exception as mex:
                        print(f"ERROR in pid {self.rank}")
                        raise mex
                elif self.rank == 0:
                    rlst = self.recieve_object_in_batches(src=i)
                    glist.append(rlst)
                    self.log_at_root(LOGGER, logging.DEBUG,
                                     "Recieved from processor :: [%d]", i)
            return glist

        @override
        def collect_merge_lists_at_root(
            self,
            in_list: list[t.Any],
            use_snd_rcv: bool = False,
        ) -> list[t.Any]:
            if self.size == 1:
                return in_list
            if use_snd_rcv:
                glist = self.gather_at_root_by_snd_rcv(in_list)
            else:
                glist = self._comm.gather(in_list)
            if glist and self.rank == 0:
                return list(itertools.chain.from_iterable(glist))
            else:
                return []

        @override
        def collect_zipmerge_lists_at_root(
            self,
            in_list: list[t.Any],
            use_snd_rcv: bool = False,
        ) -> list[t.Any]:
            if self.size == 1:
                return in_list
            if use_snd_rcv:
                glist = self.gather_at_root_by_snd_rcv(in_list)
            else:
                glist = self._comm.gather(in_list)
            if glist and self.rank == 0:
                self.log_at_root(
                    LOGGER, logging.DEBUG,
                    f"Zip Merge Lengths : {[len(x) for x in glist]}"
                )
                return list(
                    x
                    for x in itertools.chain.from_iterable(
                        itertools.zip_longest(*glist)
                    )
                    if x is not None
                )
            else:
                return []

        @override
        def check_properties_across_ranks(self,
                                          properties: dict[str, t.Any],
                                          graph_type: str='node') -> None:
            """Checks that a properties table is consistant across all MPI ranks. Mainly used by add_nodes() and add_edges()
            method due to bug where using random generator without rng_seed was causing issues building the network properties
            consistantly using multiple cores.

            Will throw an Exception or a warning message (if MPI_fail_params_nonuniform is false)

            :param properties: A dictionary
            :param graph_type: 'node' or 'edge', used in error message. default 'node'.
            """
            if self.size < 2:
                return

            # Check that model_properties have the same number of items and the keys match
            n_args = len(properties)
            ranked_args = self._comm.allgather(n_args)
            if len(set(ranked_args)) > 1:
                err_msg = '{} properties are not the same across all ranks.'.format(graph_type)
                if not MPI_fail_params_nonuniform:
                    LOGGER.warning(err_msg)
                else:
                    raise IndexError(err_msg)

            if n_args == 0:
                return

            # create a string/id that will be uniform across all ranks, even if the dict on one rank returns keys out-of-order.
            prop_keys = list(properties.keys())
            prop_keys.sort()
            combined_keys = ':'.join(prop_keys).encode('utf-8')
            hash_id = hashlib.md5(combined_keys).hexdigest()
            ranked_keys = self._comm.allgather(hash_id)
            if len(set(ranked_keys)) > 1:
                err_msg = '{} properties are not the same across all ranks.'.format(graph_type)
                if not MPI_fail_params_nonuniform:
                    LOGGER.warning(err_msg)
                else:
                    raise IndexError(err_msg)

            # For each item in model_properties dictionary try to check that values are the same
            for pkey in prop_keys:
                # Don't use Dict.items() method since it is possible the ret order is different across ranks.
                pval = properties[pkey]

                try:
                    if isinstance(pval, bytes):
                        phash = hashlib.md5(pval).hexdigest()
                    elif isinstance(pval, str):
                        phash = hashlib.md5(pval.encode('utf-8')).hexdigest()
                    elif isinstance(pval, (int, float, bool)):
                        if np.isnan(pval):
                            pval = 'NONE'
                        phash = pval
                    elif isinstance(pval, (list, tuple)):
                        joined_keys = ':'.join([str(p) for p in pval]).encode('utf-8')
                        phash = hashlib.md5(joined_keys).hexdigest()
                    elif isinstance(pval, np.ndarray):
                        phash = hashlib.md5(pval.data.tobytes()).hexdigest()
                    else:
                        continue

                except TypeError as te:
                    # If the hashing fails assume there is no MPI data issue and continue with the next property.
                    continue
                ranked_vals = self._comm.allgather(phash)
                if len(set(ranked_vals)) > 1:
                    err_msg = (
                        '{} property "{}" varies across ranks, please make sure '
                        'parameter value is uniform across all ranks or '
                        'set bmtk.builder.MPI_fail_params_nonuniform to False'
                    ).format(graph_type, pkey)
                    if not MPI_fail_params_nonuniform:
                        LOGGER.warning(err_msg)
                    else:
                        raise TypeError(err_msg)

    _default_comm_instance  = MPIComm()
    _comm_type  = MPIComm

except ImportError:
    pass


def default_comm() -> CommInterface:
    return _default_comm_instance


def instantiate_comm(**kwargs: t.Any) -> CommInterface:
    return _comm_type(**kwargs)
