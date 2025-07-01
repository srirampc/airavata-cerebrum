import datetime
import logging
import operator
import os
import psutil
import codetiming
import typing as t
import pandas as pd

ProfilesT : t.TypeAlias = list[dict[str, t.Any]]


def timestamp_prefix(rank: int):
    return f"{str(datetime.datetime.now())} :: {rank}"

def memory_profile(rank: int = 0) -> dict[str, t.Any]:
    pid = os.getpid()
    pproc = psutil.Process(pid)
    pfmem = pproc.memory_info()
    return {
        "proc": rank,
        "rss": pfmem.rss / (2**30),
        "vms": pfmem.vms / (2**30),
        "pct": pproc.memory_percent(),
    }

def timing_profile(rank: int = 0):
     rtimers = codetiming.Timer.timers
     return [{
         "proc": rank,
         "name": name,
         "ncalls": rtimers.count(name),
         "total_time": ttime,
         "min_time": rtimers.min(name),
         "max_time": rtimers.max(name),
         "mean_time": rtimers.mean(name),
         "median_time": rtimers.median(name),
         "stdev_time": rtimers.stdev(name),
     } for name, ttime in sorted(
         rtimers.items(),
         key=operator.itemgetter(1),
         reverse=True
     )]

def log_mem_usage(
    logger: logging.Logger,
    level: int,
    rank: int = 0,
):
    if not logger.isEnabledFor(level):
        return
    svmem = psutil.virtual_memory()
    used = svmem.used/(2 ** 30)
    total = svmem.total/(2 ** 30)
    avail = svmem.available/(2 ** 30)
    free = svmem.free/(2 ** 30)
    pct = svmem.percent
    logger.log(
        level,
        (
            f"{timestamp_prefix(rank)} :: "
            f"Used/Total : {used}/{total} ({pct} %); "
            f"Free : {free} ; Avail {avail}. "
            f":: CPU : {psutil.cpu_times_percent().user} %." 
         )
    )

def log_with_timestamp(
    logger: logging.Logger,
    level: int,
    message: str,
    *args: object,
    rank: int = 0,
    **kwargs: t.Any
):
    if not logger.isEnabledFor(level):
        return
    tprefix = f"{timestamp_prefix(rank)} :: "
    logger.log(level, tprefix + message, *args, **kwargs)

def log_data_frame(
    logger: logging.Logger,
    level: int,
    df: pd.DataFrame, 
    rank: int = 0,
):
    if not logger.isEnabledFor(level):
        return
    prefix = timestamp_prefix(rank)
    # set display options to show all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # print the dataframe
    for lx in df.to_string().splitlines():
        logger.log(level, prefix + lx)

