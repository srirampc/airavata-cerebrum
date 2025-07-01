import numpy as np
import logging
import typing as t
import numpy.typing as npt

from .comm_interface import default_comm, CommInterface

from .edge_props_table import MVEdgeTypesTable

logger = logging.getLogger(__name__)

NPIntArray = npt.NDArray[np.integer[t.Any]]

class MVEdgesCollator(object):
    """Used to collect all the edges data-tables created and stored in the EdgeTypesTable to simplify the process
    of saving into a SONATA edges file. All the actual edges may be stored across diffrent edge-type-tables/mpi-ranks
    and needs to be merged together (and possibly sorted) before writing to HDF5 file.
    """

    def __init__(
        self,
        edge_types_tables: list[MVEdgeTypesTable],
        network_name: str,
        distributed: bool = False,
    ):
        #
        # self._edge_types_tables: list[MVEdgeTypesTable] = edge_types_tables
        self._comm : CommInterface = default_comm()
        self._network_name: str = network_name
        self._model_groups_md: dict[int, t.Any] = {}
        self._ngroups: int = 0
        self._group_ids_lu: dict[str, t.Any] = {}
        self._group_ids: list[int] = []
        self._group_sizes : NPIntArray | None = None
        self._all_gid_counts: NPIntArray | None = None
        self._local_gid_counts: NPIntArray | None = None 
        self._group_starts:  NPIntArray | None = None
        #
        # self._grp_id_itr: int = 0
        #
        self._n_edges : int = sum(et.n_edges for et in edge_types_tables)
        self._start_index: int = 0 
        self.n_total_edges: int = self._n_edges
        self.can_sort: bool = True
        self.source_ids: NPIntArray | None = None
        self.target_ids: NPIntArray | None = None
        self.edge_type_ids: NPIntArray | None = None
        self.edge_group_ids: NPIntArray | None = None
        self.edge_group_index: NPIntArray | None = None
        self._prop_data: dict[int , t.Any] = {}
        #
        if distributed:
            raise NotImplementedError("Distributed Implemented not complete.")
            self._start_index = self._comm.counts_start_index(self._n_edges) 
            self.n_total_edges = self._comm.accumulate_counts(self._n_edges)
            self.__assign_groups_distr(edge_types_tables)
            self.__process_distr(edge_types_tables)
        else:
            self.__assign_groups_seq(edge_types_tables)
            self.__process_seq(edge_types_tables)
            self.__log();

    def __log(self):
        rshape = str(0 if self.source_ids is None else self.source_ids.shape)
        tshape = str(0 if self.target_ids is None else self.target_ids.shape)
        logger.info(f"SRC :: {rshape};; TGT :: {tshape};; GLK {self._group_ids_lu}")
        logger.info(f" MDGP {self._model_groups_md}")

    def __assign_groups_seq(self, edge_types_tables: list[MVEdgeTypesTable]):
        grp_id_itr: int = 0
        for et in edge_types_tables:
            # Assign each edge type a group_id based on the edge-type properties. When two edge-types tables use the
            # same model properties (in the hdf5) they should be put into the same group
            edge_types_hash = et.hash_key
            if edge_types_hash not in self._group_ids_lu:
                self._group_ids_lu[edge_types_hash] = grp_id_itr

                group_metadata = et.get_property_metadata()
                self._model_groups_md[grp_id_itr] = {
                    'prop_names': [p['name'] for p in group_metadata],
                    'prop_type': [p['dtype'] for p in group_metadata],
                    'prop_size': 0
                }
                grp_id_itr += 1

            group_id = self._group_ids_lu[edge_types_hash]
            et.edge_group_id = group_id

            # number of rows in each model group
            self._model_groups_md[group_id]['prop_size'] += et.n_edges

    def __process_seq(self, edge_types_tables: list[MVEdgeTypesTable]):
        logger.debug('Processing and collating {:,} edges.'.format(self.n_total_edges))

        self.source_ids = np.zeros(self.n_total_edges, dtype=np.uint)
        self.target_ids = np.zeros(self.n_total_edges, dtype=np.uint)
        self.edge_type_ids = np.zeros(self.n_total_edges, dtype=np.uint32)
        self.edge_group_ids = np.zeros(self.n_total_edges, dtype=np.uint32)
        self.edge_group_index = np.zeros(self.n_total_edges, dtype=np.uint32)

        self._prop_data = {
            g_id: {
                n: np.zeros(g_md['prop_size'], dtype=t) for n, t in zip(g_md['prop_names'], g_md['prop_type'])
            } for g_id, g_md in self._model_groups_md.items()
        }

        idx_beg = 0
        group_idx = {g_id: 0 for g_id in self._model_groups_md.keys()}
        for et in edge_types_tables:
            idx_end = idx_beg + et.n_edges

            src_trg_ids = et.edge_type_node_ids
            self.source_ids[idx_beg:idx_end] = src_trg_ids[:, 0]
            self.target_ids[idx_beg:idx_end] = src_trg_ids[:, 1]
            self.edge_type_ids[idx_beg:idx_end] = et.edge_type_id
            self.edge_group_ids[idx_beg:idx_end] = et.edge_group_id

            group_idx_beg = group_idx[et.edge_group_id]
            group_idx_end = group_idx_beg + et.n_edges
            self.edge_group_index[idx_beg:idx_end] = np.arange(group_idx_beg, group_idx_end, dtype=np.uint32)
            for pname, pdata in self._prop_data[et.edge_group_id].items():
                pdata[group_idx_beg:group_idx_end] = et.get_property_value(pname)

            idx_beg = idx_end
            group_idx[et.edge_group_id] = group_idx_end

            et.free_data()

    def __assign_groups_distr(self, edge_types_tables: list[MVEdgeTypesTable]):
        # get unique has and its meta data
        grp_hash_keys = sorted(set([et.hash_key for et in edge_types_tables]))
        grp_hash_keys = self._comm.collect_merge_lists(grp_hash_keys)
        srt_hash_keys = sorted(set(grp_hash_keys)) if grp_hash_keys else []
        self._ngroups = len(srt_hash_keys)
        self._group_ids_lu  = dict(zip(srt_hash_keys, range(self._ngroups)))
        self._group_starts = np.zeros(self._ngroups, dtype=np.uint32)
        self._local_gid_counts = np.zeros(self._ngroups, dtype=np.uint32)
        for ret in edge_types_tables:
            g_id = self._group_ids_lu[ret.hash_key]
            # number of rows in each model group
            self._local_gid_counts[g_id] += ret.n_edges
        self._all_gid_counts, self._group_sizes = self._comm.gather_np_counts_2d(
            self._local_gid_counts
        )
        self._group_starts[0] = 0
        #
        for ix in range(1, self._ngroups):
            self._group_starts[0] = self._group_starts[ix-1] + self._group_sizes[ix-1]
        #
        for ret in edge_types_tables:
            # Assign each edge type a group_id based on the edge-type properties.
            # When two edge-types tables use the same
            # model properties (in the hdf5) they should be
            # put into the same group
            edge_types_hash = ret.hash_key
            g_id = self._group_ids_lu[edge_types_hash]
            if g_id not in self._model_groups_md:
                group_metadata = ret.get_property_metadata()
                self._model_groups_md[g_id] = {
                    'prop_names': [p['name'] for p in group_metadata],
                    'prop_type': [p['dtype'] for p in group_metadata],
                    'prop_size': self._group_sizes[g_id]
                }

    def __process_distr(self, edge_types_tables: list[MVEdgeTypesTable]):
        logger.debug('Processing and collating {:,} edges.'.format(self.n_total_edges))
        self.source_ids = np.zeros(self._n_edges, dtype=np.uint)
        self.target_ids = np.zeros(self._n_edges, dtype=np.uint)
        self.edge_type_ids = np.zeros(self._n_edges, dtype=np.uint32)
        self.edge_group_ids = np.zeros(self._n_edges, dtype=np.uint32)
        self.edge_group_index = np.zeros(self._n_edges, dtype=np.uint32)

        self._prop_data = {
            g_id: {
                n: np.zeros(g_md['prop_size'], dtype=t)
                for n, t in zip(g_md['prop_names'], g_md['prop_type'])
            } for g_id, g_md in self._model_groups_md.items()
        }

        ridx_beg = 0
        for ret in edge_types_tables:
            edge_types_hash = ret.hash_key
            et_group_id = self._group_ids_lu[edge_types_hash]
            src_trg_ids = ret.edge_type_node_ids
            ridx_end = ridx_beg + ret.n_edges
            self.source_ids[ridx_beg:ridx_end] = src_trg_ids[:, 0]
            self.target_ids[ridx_beg:ridx_end] = src_trg_ids[:, 1]
            self.edge_type_ids[ridx_beg:ridx_end] = ret.edge_type_id
            self.edge_group_ids[ridx_beg:ridx_end] = et_group_id
            ridx_beg = ridx_end
        #
        # TODO: fix this later
        # group_idx = np.zeros(self._ngroups, dtype=np.uint32)
        # for ix in range(1, self._ngroups):
        #     group_idx[ix] = group_idx[ix - 1] + self._local_gid_counts[ix - 1]
        # for ret in edge_types_tables:
        #     edge_types_hash = ret.hash_key
        #     et_group_id = self._group_ids_lu[edge_types_hash]
        #     src_trg_ids = ret.edge_type_node_ids

        #     group_idx_beg = group_idx[et_group_id]
        #     group_idx_end = group_idx_beg + ret.n_edges
        #     for pname, pdata in self._prop_data[et_group_id].items():
        #         pdata[group_idx_beg:group_idx_end] = ret.get_property_value(pname)
        #     group_idx[et_group_id] += ret.n_edges 
        #
        # TODO:: fix
        # for ret in edge_types_tables:
        #    self.edge_group_index[idx_beg:idx_end] = np.arange(group_idx_beg,
        #                                                       group_idx_end,
        #                                                       dtype=np.uint32)


        for ret in edge_types_tables:
            ret.free_data()


    # @property
    # def group_ids(self):
    #     return self._group_ids
    @property
    def group_ids(self):
        return list(self._prop_data.keys())


    def get_group_metadata(self, group_id: int):
        grp_md = self._model_groups_md[group_id]
        grp_dim = (grp_md['prop_size'], )
        return [
            {'name': n, 'type': t, 'dim': grp_dim}
            for n, t in zip(grp_md['prop_names'], grp_md['prop_type'])
        ]

    def itr_chunks(self):
        chunk_id = 0
        idx_beg = 0
        idx_end = self.n_total_edges

        yield chunk_id, idx_beg, idx_end

    def get_target_node_ids(self, chunk_id: int):
        return self.target_ids

    def get_source_node_ids(self, chunk_id: int):
        return self.source_ids

    def get_edge_type_ids(self, chunk_id: int):
        return self.edge_type_ids

    def get_edge_group_ids(self, chunk_id: int):
        return self.edge_group_ids

    def get_edge_group_indices(self, chunk_id: int):
        return self.edge_group_index

    def get_group_data(self, chunk_id: int):
        ret_val = []
        for group_id in self._prop_data.keys():
            for group_name in self._prop_data[group_id].keys():

                idx_end = len(self._prop_data[group_id][group_name])
                ret_val.append((group_id, group_name, 0, idx_end))

        return ret_val

    def get_group_property(self, group_name: str, group_id: int, chunk_id: int):
        return self._prop_data[group_id][group_name]

    def sort(self, sort_by: str, sort_group_properties: bool=True):
        """In memory sort of the dataset

        :param sort_by:
        :param sort_group_properties:
        """
        # Find the edges hdf5 column to sort by
        if sort_by == 'target_node_id':
            sort_ds = self.target_ids
        elif sort_by == 'source_node_id':
            sort_ds = self.source_ids
        elif sort_by == 'edge_type_id':
            sort_ds = self.edge_type_ids
        elif sort_by == 'edge_group_id':
            sort_ds = self.edge_group_ids
        else:
            logger.warning('Unable to sort dataset, unrecognized column {}.'.format(sort_by))
            return

        # check if dataset is already sorted
        if sort_ds is None or np.all(np.diff(sort_ds) <= 0):
            return

        # Find order of arguments of sorted arrays, and sort main columns
        sort_idx = np.argsort(sort_ds)
        self.source_ids = self.source_ids[sort_idx]
        self.target_ids = self.target_ids[sort_idx]
        self.edge_type_ids = self.edge_type_ids[sort_idx]
        self.edge_group_ids = self.edge_group_ids[sort_idx]
        self.edge_group_index = self.edge_group_index[sort_idx]

        if sort_group_properties:
            # For sorting group properties, so the "edge_group_index" column is sorted (wrt each edge_group_id). Note
            # that it is not strictly necessary to sort the group properties, as sonata edge_group_index keeps the
            # reference, but doing the sorting may improve memory/efficency during setting up a simulation

            for grp_id, grp_props in self._prop_data.items():
                # Filter out edge_group_index array for each group_id, get the new order and apply to each property.
                grp_id_filter = np.argwhere(self.edge_group_ids == grp_id).flatten()
                updated_order = self.edge_group_index[grp_id_filter]
                for prop_name, prop_data in grp_props.items():
                    grp_props[prop_name] = prop_data[updated_order]

                # reorder the edge_group_index (for values with corresponding group_id)
                self.edge_group_index[grp_id_filter] = np.arange(0, len(grp_id_filter), dtype=np.uint32)
