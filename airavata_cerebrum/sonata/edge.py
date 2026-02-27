"""
SONATA Edge File Editor
=======================
Context-manager class for reading and modifying SONATA edge HDF5 files.

Basic usage
-----------
>>> from sonata_edge_editor import SonataEdgeEditor, EditConfig, IndexType
>>>
>>> cfg = EditConfig(
...     edge_type_index_group="edge_type_to_index",
...     type_to_range_dataset="node_id_to_range",
...     range_to_edge_dataset="range_to_edge_id",
... )
>>>
>>> with SonataEdgeEditor("edges.h5", edge_type_index_cfg=cfg) as editor:
...     print(editor.has_indices())
...     edges = editor.get_edges_by_type(100)
...     new_edges = edges.drop("_row_index")   # modify as needed
...     # single type
...     editor.replace_edges_by_type(100, new_edges, output_path="edges_modified.h5")
...     # multiple types in one pass
...     editor.replace_edges_by_multiple_types({100: new_edges, 200: new_edges}, output_path="edges_modified.h5")

SONATA HDF5 layout assumed
--------------------------
edges/<population>/
    source_node_id      [N]
    target_node_id      [N]
    edge_type_id        [N]
    edge_group_id       [N]          (optional)
    edge_group_index    [N]          (optional)
    0/                               (per-edge properties)
        <prop>          [M] ...
    indices/                         (optional)
        source_to_target/            <- standard SONATA node index
            node_id_to_ranges   [n_source_nodes x 2]
            range_to_edge_id    [n_ranges x 2]
        target_to_source/            <- standard SONATA node index
            node_id_to_ranges   [n_target_nodes x 2]
            range_to_edge_id    [n_ranges x 2]
        <edge_type_group>/           <- optional non-standard edge-type index
            <type_to_range_ds>  [n_edge_types x 2]
            <range_to_edge_ds>  [n_ranges x 2]
"""

import logging
import shutil
import typing as t
from enum import Flag, auto
from pathlib import Path
from types import TracebackType

import h5py
import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, computed_field
from typing_extensions import override

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HDF5 path constants  (refactor #4)
# Used in both has_indices() and _do_rebuild_indices() to avoid duplication.
# ---------------------------------------------------------------------------

_S2T_N2R = "indices/source_to_target/node_id_to_ranges"
_S2T_R2E = "indices/source_to_target/range_to_edge_id"
_T2S_N2R = "indices/target_to_source/node_id_to_ranges"
_T2S_R2E = "indices/target_to_source/range_to_edge_id"

_NODE_INDEX_PATHS: tuple[str, ...] = (_S2T_N2R, _S2T_R2E, _T2S_N2R, _T2S_R2E)


# ---------------------------------------------------------------------------
# IndexType  (Flag enum)
# ---------------------------------------------------------------------------


class IndexType(Flag):
    """
    Controls which index tables are built or checked.

    Members can be combined with ``|``::

        IndexType.NODE | IndexType.EDGE_TYPE  # same as IndexType.ALL
    """

    NODE = auto()  # standard source_to_target / target_to_source
    EDGE_TYPE = auto()  # non-standard edge-type index
    ALL = NODE | EDGE_TYPE


# ---------------------------------------------------------------------------
# EditConfig  (pydantic BaseModel)
# ---------------------------------------------------------------------------


class EditConfig(BaseModel):
    """
    Describes the layout of a non-standard edge-type index inside
    ``edges/<population>/indices/``.

    Parameters
    ----------
    edge_type_index_group:
        HDF5 group name under ``indices/``.
        Example: ``"edge_type_to_index"``.
    type_to_range_dataset:
        Dataset mapping edge_type_id -> range rows.  Shape [n_types x 2].
        Example: ``"node_id_to_range"`` (Blue Brain naming, likely a typo).
    range_to_edge_dataset:
        Dataset holding actual edge-row ranges.  Shape [n_ranges x 2].
        Almost always ``"range_to_edge_id"``.

    Example
    -------
    >>> cfg = EditConfig(
    ...     edge_type_index_group="edge_type_to_index",
    ...     type_to_range_dataset="node_id_to_range",
    ...     range_to_edge_dataset="range_to_edge_id",
    ... )
    """

    edge_type_index_group: str
    type_to_range_dataset: str = "node_id_to_range"
    range_to_edge_dataset: str = "range_to_edge_id"

    @computed_field
    @property
    def type_to_range_path(self) -> str:
        return f"indices/{self.edge_type_index_group}/{self.type_to_range_dataset}"

    @computed_field
    @property
    def range_to_edge_path(self) -> str:
        return f"indices/{self.edge_type_index_group}/{self.range_to_edge_dataset}"


# ---------------------------------------------------------------------------
# SonataEdgeEditor
# ---------------------------------------------------------------------------


class SonataEdgeEditor:
    """
    Context manager for reading and modifying a SONATA edge HDF5 file.

    Opens the file in read-only mode on entry so that inspection methods
    (``get_edges_by_type``, ``has_indices``) work without any risk of
    accidental writes.  ``replace_edges_by_type`` and ``rebuild_indices``
    operate on a copy / backup and close-then-reopen as needed.

    Parameters
    ----------
    h5_path:
        Path to the SONATA edges HDF5 file.
    population:
        Edge population name.  If None, the first population found is used
        and stored on ``self.population`` after opening.
    edge_type_index_cfg:
        Configuration for the non-standard edge-type index.  Pass this
        when your file contains an edge-type index with non-standard names.

    Examples
    --------
    Read-only inspection::

        with SonataEdgeEditor("edges.h5") as editor:
            print(editor.population)
            print(editor.has_indices())
            df = editor.get_edges_by_type(100)

    Edit and write to a new file::

        cfg = EditConfig(
            edge_type_index_group="edge_type_to_index",
            type_to_range_dataset="node_id_to_range",
        )

        with SonataEdgeEditor("edges.h5", edge_type_index_cfg=cfg) as editor:
            edges = editor.get_edges_by_type(100)
            new_edges = edges.drop("_row_index")  # ... your modifications ...
            editor.replace_edges_by_type(
                edge_type_id=100,
                new_edges=new_edges,
                output_path="edges_modified.h5",
            )

    In-place edit (auto-backup created)::

        with SonataEdgeEditor("edges.h5") as editor:
            edges = editor.get_edges_by_type(100)
            editor.replace_edges_by_type(100, edges.drop("_row_index"))
    """

    # Names treated as top-level index arrays, never as per-edge properties
    _TOP_LEVEL: frozenset[str] = frozenset(
        [
            "source_node_id",
            "target_node_id",
            "edge_type_id",
            "edge_group_id",
            "edge_group_index",
        ]
    )

    def __init__(
        self,
        h5_path: str | Path,
        population: str | None = None,
        edge_type_index_cfg: EditConfig | None = None,
    ) -> None:
        self._h5_path: Path = Path(h5_path)
        self._requested_population: str | None = population
        self.edge_type_index_cfg: EditConfig | None = edge_type_index_cfg

        # set after __enter__
        self.population: str = ""
        self._h5: h5py.File | None = None

    # ------------------------------------------------------------------ #
    # Context manager protocol
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "SonataEdgeEditor":
        self._h5 = h5py.File(self._h5_path, "r")
        self.population = self._resolve_population(self._h5, self._requested_population)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_edges_by_type(self, edge_type_id: int) -> pl.DataFrame:
        """
        Return all edges of the given edge_type_id as a Polars DataFrame.

        Parameters
        ----------
        edge_type_id:
            The edge type ID to filter on.

        Returns
        -------
        pl.DataFrame
            All matching rows plus a ``_row_index`` column with original
            row positions.

        Raises
        ------
        RuntimeError
            If called outside a ``with`` block.
        """
        self._require_open()
        grp = self._population_group()
        df = self._read_population_to_df(grp).with_row_index("_row_index")
        filtered = df.filter(pl.col("edge_type_id") == edge_type_id)
        if filtered.is_empty():
            log.warning("No edges found with edge_type_id=%d", edge_type_id)
        return filtered

    def has_indices(self) -> IndexType | None:
        """
        Check which index tables are present in the edge population.

        Returns
        -------
        IndexType | None
            NODE        # standard source_to_target / target_to_source
            EDGE_TYPE   # non-standard edge-type index
            ALL         # both NODE and EDGE_TYPE
        """
        self._require_open()
        grp = self._population_group()

        node_ok = all(p in grp for p in _NODE_INDEX_PATHS)

        type_ok = False
        if self.edge_type_index_cfg is not None:
            cfg = self.edge_type_index_cfg
            type_ok = cfg.type_to_range_path in grp and cfg.range_to_edge_path in grp

        if node_ok and type_ok:
            return IndexType.ALL
        if node_ok:
            return IndexType.NODE
        if type_ok:
            return IndexType.EDGE_TYPE

    def replace_edges_by_type(
        self,
        edge_type_id: int,
        new_edges: pl.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Replace all edges of ``edge_type_id`` with ``new_edges``.

        Convenience wrapper around :meth:`replace_edges_by_multiple_types`
        for the common single-type case.  See that method for full parameter
        documentation.
        """
        return self.replace_edges_by_multiple_types(
            {edge_type_id: new_edges},
            output_path=output_path,
        )

    def replace_edges_by_multiple_types(
        self,
        replacements: dict[int, pl.DataFrame],
        output_path: str | Path | None = None,
    ) -> Path:
        """
        Replace edges for multiple edge types in a single file pass.

        More efficient than calling :meth:`replace_edges_by_type` in a loop
        because the file is copied once, read once, and indices are rebuilt
        once regardless of how many types are being replaced.

        Parameters
        ----------
        replacements:
            Mapping of ``edge_type_id -> replacement DataFrame``.  Each
            DataFrame must contain at least ``source_node_id``,
            ``target_node_id``, and ``edge_type_id``.  ``_row_index``
            columns are silently dropped.  Schemas may differ across types;
            missing columns are null-filled via ``diagonal`` concat.
        output_path:
            Destination file.  If None, the source file is edited in-place
            after creating a ``.bak`` backup.

        Returns
        -------
        Path
            Path to the file that was written.

        Raises
        ------
        RuntimeError
            If called outside a ``with`` block.
        ValueError
            If ``replacements`` is empty.
        """
        self._require_open()

        if not replacements:
            raise ValueError("replacements dict must contain at least one entry.")

        # drop _row_index from all incoming frames up front
        cleaned: dict[int, pl.DataFrame] = {
            eid: df.drop("_row_index") if "_row_index" in df.columns else df
            for eid, df in replacements.items()
        }

        # snapshot index status before closing the read handle
        index_avail = self.has_indices()

        # resolve output / backup
        if output_path is None:
            backup = self._h5_path.with_suffix(".h5.bak")
            shutil.copy2(self._h5_path, backup)
            log.info("Backup created: %s", backup)
            out_path = self._h5_path
        else:
            out_path = Path(output_path)
            shutil.copy2(self._h5_path, out_path)

        # close read handle so we can open for writing
        self._h5.close()
        self._h5 = None

        try:
            self._write_replacements(out_path, cleaned)
            self._rebuild_as_needed(out_path, index_avail)
        finally:
            # reopen in read mode so the editor remains usable after the call
            self._h5 = h5py.File(self._h5_path, "r")

        return out_path

    def rebuild_indices(
        self,
        target_path: str | Path | None = None,
        which: IndexType = IndexType.ALL,
    ) -> None:
        """
        Rebuild index tables on a file.

        Parameters
        ----------
        target_path:
            File to rebuild indices on.  Defaults to the editor's source
            file.  The file is modified in-place.
        which:
            Which index tables to rebuild.  Default: ``IndexType.ALL``.
            Examples::

                IndexType.NODE
                IndexType.EDGE_TYPE
                IndexType.NODE | IndexType.EDGE_TYPE  # same as ALL
        """
        if IndexType.EDGE_TYPE in which and self.edge_type_index_cfg is None:
            raise ValueError(
                "edge_type_index_cfg must be set on the editor when IndexType.EDGE_TYPE is requested"
            )

        path = Path(target_path) if target_path is not None else self._h5_path

        # must not have the file open read-only while we write it
        was_open = self._h5 is not None
        if was_open and path == self._h5_path:
            self._h5.close()
            self._h5 = None

        try:
            self._do_rebuild_indices(path, which)
        finally:
            if was_open and path == self._h5_path:
                self._h5 = h5py.File(self._h5_path, "r")

    # ------------------------------------------------------------------ #
    # Internal helpers — HDF5 access
    # ------------------------------------------------------------------ #

    def _require_open(self) -> None:
        if self._h5 is None:
            raise RuntimeError(
                "SonataEdgeEditor must be used as a context manager (use 'with SonataEdgeEditor(...) as editor:')"
            )

    @staticmethod
    def _edge_keys(h5: h5py.File | None) -> list[str]:
        return list(t.cast(h5py.Group, h5.get("edges", {})).keys()) if h5 else []

    @staticmethod
    def _resolve_population(h5: h5py.File, population: str | None) -> str:
        if population is not None:
            return population
        keys = SonataEdgeEditor._edge_keys(h5)
        if not keys:
            raise KeyError("No edge populations found in file.")
        return keys[0]

    def _population_group(self, h5: h5py.File | None = None) -> h5py.Group:
        f = h5 if h5 is not None else self._h5
        try:
            return t.cast(h5py.Group, f[f"edges/{self.population}"])
        except KeyError:
            available = SonataEdgeEditor._edge_keys(f)
            raise KeyError(
                f"Population '{self.population}' not found. Available: {available}"
            )

    def _read_population_to_df(self, grp: h5py.Group) -> pl.DataFrame:
        """Read all edge data (skipping indices/) into a Polars DataFrame."""
        data: dict[str, NDArray[t.Any]] = {}

        for key in self._TOP_LEVEL:
            if key in grp:
                pds = t.cast(h5py.Dataset, grp[key])
                data[key] = pds[:]

        n_edges = len(data.get("source_node_id", []))

        for subkey in grp.keys():
            if not subkey.isdigit():
                continue
            sub_grp = t.cast(h5py.Group, grp[subkey])
            group_id = int(subkey)
            if "edge_group_id" in data and "edge_group_index" in data:
                mask = data["edge_group_id"] == group_id
                indices = data["edge_group_index"][mask]
            else:
                mask = np.ones(n_edges, dtype=bool)
                indices = np.arange(n_edges)

            for prop in sub_grp.keys():
                if isinstance(sub_grp[prop], h5py.Dataset):
                    pds = t.cast(h5py.Dataset, sub_grp[prop])
                    col = np.full(
                        n_edges,
                        fill_value=np.nan if pds.dtype.kind == "f" else 0,
                        dtype=pds.dtype,
                    )
                    col[mask] = pds[indices]
                    data[prop] = col

        return pl.DataFrame(data)

    # ------------------------------------------------------------------ #
    # Internal helpers — writing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_edge_groups(grp: h5py.Group) -> int:
        """Return the number of numeric sub-groups (edge property groups)."""
        return sum(1 for k in grp.keys() if k.isdigit())

    def _write_replacements(
        self,
        out_path: Path,
        replacements: dict[int, pl.DataFrame],
    ) -> None:
        with h5py.File(out_path, "r+") as h5:
            grp = self._population_group(h5)

            # Guard: multi-group files require per-group edge_group_index
            # logic that this editor does not yet support.
            n_groups = self._count_edge_groups(grp)
            if n_groups > 1:
                raise NotImplementedError(
                    f"Population '{self.population}' has {n_groups} edge property groups. This editor only supports single-group populations. Multi-group support requires per-group edge_group_index regeneration, which is not yet implemented."
                )

            # Single read of the full population; filter out all replaced types at once.
            existing = self._read_population_to_df(grp).filter(
                ~pl.col("edge_type_id").is_in(list(replacements.keys()))
            )

            merged = pl.concat([existing, *replacements.values()], how="diagonal").sort(
                ["edge_type_id", "source_node_id"]
            )
            n_new = len(merged)

            # Regenerate edge_group_id and edge_group_index for the
            # single-group case. After sort+concat the row positions have
            # changed, so any previously stored values are stale.
            # edge_group_id    = 0 for every edge (single group "0")
            # edge_group_index = row position within that group, which in
            # the single-group case is simply 0, 1, 2, ... N-1.
            merged = merged.with_columns(
                [
                    pl.lit(0).cast(pl.Int32).alias("edge_group_id"),
                    pl.int_range(0, n_new, dtype=pl.Int64).alias("edge_group_index"),
                ]
            )

            for col in self._TOP_LEVEL:
                if col in merged.columns and col in grp:
                    del grp[col]
                    grp.create_dataset(col, data=merged[col].to_numpy())
                elif col in grp:
                    ds = t.cast(h5py.Dataset, grp[col])
                    arr = ds[:]
                    del grp[col]
                    pad = np.zeros(n_new, dtype=arr.dtype)
                    pad[: min(len(arr), n_new)] = arr[: min(len(arr), n_new)]
                    grp.create_dataset(col, data=pad)

            prop_cols = [c for c in merged.columns if c not in self._TOP_LEVEL]
            if "0" not in grp:
                grp.create_group("0")
            sub = t.cast(h5py.Group, grp["0"])
            for col in prop_cols:
                if col in sub:
                    del sub[col]
                sub.create_dataset(col, data=merged[col].to_numpy())
            for col in list(sub.keys()):
                if col not in prop_cols:
                    log.info("Removing obsolete property dataset '0/%s'", col)
                    del sub[col]

        _replaced_counts = {eid: len(df) for eid, df in replacements.items()}
        log.info(
            "Edges written: population '%s' now has %d edges. Replaced types: %s.",
            self.population,
            n_new,
            _replaced_counts,
        )

    def _rebuild_as_needed(self, out_path: Path, which: IndexType | None) -> None:
        if which is None:
            log.info("No index tables in source file — skipping rebuild.")
            return
        log.info("Rebuilding index tables …")
        #which = IndexType(0)
        #if need_node:
        #    which |= IndexType.NODE
        #if need_type:
        #    which |= IndexType.EDGE_TYPE
        self._do_rebuild_indices(out_path, which)

    def _rebuild_node_indices(self, grp: h5py.Group):
        source_ids = t.cast(h5py.Dataset, grp["source_node_id"])[:]
        target_ids = t.cast(h5py.Dataset, grp["target_node_id"])[:]
        src_order = np.argsort(source_ids, kind="stable")
        s2t_n2r, s2t_r2e = self._build_node_index_arrays(
            source_ids[src_order], src_order
        )
        tgt_order = np.argsort(target_ids, kind="stable")
        t2s_n2r, t2s_r2e = self._build_node_index_arrays(
            target_ids[tgt_order], tgt_order
        )

        self._write_ds(grp, _S2T_N2R, s2t_n2r)
        self._write_ds(grp, _S2T_R2E, s2t_r2e)
        self._write_ds(grp, _T2S_N2R, t2s_n2r)
        self._write_ds(grp, _T2S_R2E, t2s_r2e)
        log.info(
            "Node index rebuilt: %d source nodes, %d target nodes.",
            len(s2t_n2r),
            len(t2s_n2r),
        )

    def _rebuild_edge_index(self, grp: h5py.Group):
        edge_type_ids = t.cast(h5py.Dataset, grp["edge_type_id"])[:]
        cfg = self.edge_type_index_cfg
        sorted_type_ids = np.sort(edge_type_ids)
        type_to_range, r2e = self._build_edge_type_index_arrays(sorted_type_ids)
        self._write_ds(grp, cfg.type_to_range_path, type_to_range)
        self._write_ds(grp, cfg.range_to_edge_path, r2e)
        log.info(
            "Edge-type index rebuilt: %d type slots, %d ranges"
            + " (group: '%s', lookup dataset: '%s').",
            len(type_to_range),
            len(r2e),
            cfg.edge_type_index_group,
            cfg.type_to_range_dataset,
        )

    def _do_rebuild_indices(self, path: Path, which: IndexType) -> None:
        with h5py.File(path, "r+") as h5:
            grp = self._population_group(h5)
            n_edges = len(t.cast(h5py.Dataset, grp["source_node_id"]))

            if IndexType.NODE in which:
                self._rebuild_node_indices(grp)

            if IndexType.EDGE_TYPE in which and self.edge_type_index_cfg is not None:
                self._rebuild_edge_index(grp)
            elif IndexType.EDGE_TYPE in which and self.edge_type_index_cfg is None:
                log.warning(
                    "Edge-type index requested but no config provided — skipped."
                )

        log.info(
            "rebuild_indices done for population '%s' (%d edges).",
            self.population,
            n_edges,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers — index array builders (static)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_ds(grp: h5py.Group, path: str, data: NDArray[t.Any]) -> None:
        if path in grp:
            del grp[path]
        grp.create_dataset(path, data=data)

    @staticmethod
    def _build_node_index_arrays(
        node_ids: NDArray[np.int64],
        edge_indices: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Build node_id_to_ranges and range_to_edge_id for one direction.

        Fully vectorised — O(N) with no Python-level loops.

        Parameters
        ----------
        node_ids:
            Per-edge node-ID array sorted ascending (after argsort).
        edge_indices:
            Global edge row indices corresponding to the sorted node_ids.

        How it works
        ------------
        range_to_edge_id  : each row is [edge_idx, edge_idx+1], so we just
                            stack edge_indices with edge_indices+1. Shape [N x 2].

        node_id_to_ranges : np.unique with return_counts gives us each unique
                            node ID and how many edges it has. A cumsum over
                            the counts gives the end-of-run positions in
                            range_to_edge_id; shifting by one gives the starts.
                            Nodes with no edges keep their -1 sentinel.
        """
        if len(node_ids) == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

        # range_to_edge_id — one row per edge, no Python loop needed
        r2e = np.stack([edge_indices, edge_indices + 1], axis=1).astype(np.int64)

        # node_id_to_ranges — derive run boundaries from unique counts
        unique_ids, counts = np.unique(node_ids, return_counts=True)
        ends = np.cumsum(counts)
        starts = ends - counts

        n2r = np.full((int(node_ids.max()) + 1, 2), -1, dtype=np.int64)
        n2r[unique_ids, 0] = starts
        n2r[unique_ids, 1] = ends

        return n2r, r2e

    @staticmethod
    def _build_edge_type_index_arrays(
        edge_type_ids: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Build type_to_range and range_to_edge_id for the edge-type index.

        Fully vectorised — O(N) with no Python-level loops.

        The array must be sorted by edge_type_id before calling.

        How it works
        ------------
        Each unique type maps to exactly one contiguous range of edge rows,
        so range_to_edge_id has one row per unique type.  np.unique with
        return_counts gives us the run lengths; a cumsum turns those into
        the [start, end) pairs directly.
        """
        if len(edge_type_ids) == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

        unique_ids, counts = np.unique(edge_type_ids, return_counts=True)
        ends = np.cumsum(counts)
        starts = ends - counts

        # range_to_edge_id — one contiguous [start, end) per unique type
        r2e = np.stack([starts, ends], axis=1).astype(np.int64)

        # type_to_range — maps type_id -> row index in r2e
        t2r = np.full((int(edge_type_ids.max()) + 1, 2), -1, dtype=np.int64)
        range_positions = np.arange(len(unique_ids), dtype=np.int64)
        t2r[unique_ids, 0] = range_positions
        t2r[unique_ids, 1] = range_positions + 1

        return t2r, r2e

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    @override
    def __repr__(self) -> str:
        state = "open" if self._h5 is not None else "closed"
        return (
            f"SonataEdgeEditor("
            f"path='{self._h5_path}', "
            f"population='{self.population}', "
            f"state={state})"
        )
