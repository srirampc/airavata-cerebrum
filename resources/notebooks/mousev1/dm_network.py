# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# from _typeshed import ConvertibleToInt
import os
import numpy as np
import h5py
import logging
import typing as t

import pandas as pd

from typing_extensions import override
from collections.abc import Iterable

from bmtk.builder.node_pool import NodePool
from bmtk.builder.node_set import NodeSet
from bmtk.builder.node import Node
from bmtk.builder.edge import Edge
from bmtk.utils import sonata

from bmtk.builder.index_builders import create_index_in_memory
from bmtk.builder.builder_utils import mpi_rank, mpi_size, barrier
from bmtk.builder.edges_sorter import sort_edges
from bmtk.builder.connection_map import ConnectionMap

from .mpi_utils import (
    accumulate_counts,
    collect_counts,
    collect_merge_lists,
    collect_zipmerge_lists_at_root,
    log_at_root,
)

from .edge_props_table import MVEdgeTypesTable
from .network import MVNetwork
from .edge_collator import MVEdgesCollator

from codetiming import Timer

logger = logging.getLogger(__name__)


def add_hdf5_attrs(hdf5_handle: h5py.File):
    # TODO: move this as a utility function
    hdf5_handle['/'].attrs['magic'] = np.uint32(0x0A7A)
    hdf5_handle['/'].attrs['version'] = [np.uint32(0), np.uint32(1)]


@Timer(name="_sort_on_disk", logger=None) 
def _sort_on_disk(
    pop_name: str,
    edges_file_name: str,
    edges_file_name_final: str,
    sort_by: str | None,
    compression: str | None
): 
    logger.debug('Sorting {} by {} to {}'.format(
        edges_file_name, sort_by, edges_file_name_final))
    sort_edges(
        input_edges_path=edges_file_name,
        output_edges_path=edges_file_name_final,
        edges_population='/edges/{}'.format(pop_name),
        sort_by=sort_by,
        compression=compression  # pyright: ignore[reportArgumentType]
        # sort_on_disk=True,
    )
    try:
        logger.debug('Deleting intermediate edges file {}.'.format(edges_file_name))
        os.remove(edges_file_name)
    except OSError as e:  # pragma: no cover
        logger.warning('Unable to remove intermediate edges file {}.'.format(edges_file_name)) 


@Timer(name="_index_on_disk", logger=None) 
def _index_on_disk(
    index_by: list[t.Any] | tuple[t.Any] | object,
    pop_name: str,
    edges_file_name_final: str,
    compression: str | None
):
    index_by = index_by if isinstance(index_by, (list, tuple)) else [index_by]
    for index_type in index_by:
        logger.debug('Creating index {}'.format(index_type))
        create_index_in_memory(
            edges_file=edges_file_name_final,
            edges_population='/edges/{}'.format(pop_name),
            index_type=index_type,
            compression=compression  # pyright: ignore[reportArgumentType]
        )

@t.final
class NodesCollator:
    def __init__(self, nnodes: int) -> None:
        self.group_props = {}
        self.node_gid_table = np.zeros(nnodes)  # todo: set dtypes
        self.node_type_id_table = np.zeros(nnodes)
        self.node_group_table = np.zeros(nnodes)
        self.node_group_index_tables = np.zeros(nnodes)

    def build_from_nodesets(self, node_sets: list[NodeSet], node_pool: NodePool):
        # save the node_types file
        group_indx = 0
        groups_lookup = {}
        group_indicies = {}
        self.group_props = {}
        for ns in node_sets:
            if ns.params_hash in groups_lookup:
                continue
            else:
                groups_lookup[ns.params_hash] = group_indx
                group_indicies[group_indx] = 0
                self.group_props[group_indx] = {k: [] for k in ns.params_keys if k != 'node_id'}
                group_indx += 1

        for i, node in enumerate(node_pool):
            self.node_gid_table[i] = node.node_id
            self.node_type_id_table[i] = node.node_type_id
            group_id = groups_lookup[node.params_hash]
            self.node_group_table[i] = group_id
            self.node_group_index_tables[i] = group_indicies[group_id]
            group_indicies[group_id] += 1
            group_dict = self.group_props[group_id]
            for key, prop_ds in group_dict.items():
                prop_ds.append(node.params[key])


@Timer(name="_write_nodes_h5", logger=None) 
def _write_nodes_h5(
    nsh: NodesCollator,
    name: str,
    nodes_file_name: str,
    mode: str,
    compression:str | None,
):
     with h5py.File(nodes_file_name, mode) as hf:
         # Add magic and version attribute
         add_hdf5_attrs(hf)
         pop_grp = hf.create_group('/nodes/{}'.format(name))
         pop_grp.create_dataset('node_id', data=nsh.node_gid_table, dtype='uint64', compression=compression)
         pop_grp.create_dataset('node_type_id', data=nsh.node_type_id_table, dtype='uint64', compression=compression)
         pop_grp.create_dataset('node_group_id', data=nsh.node_group_table, dtype='uint32', compression=compression)
         pop_grp.create_dataset('node_group_index', data=nsh.node_group_index_tables, dtype='uint64', compression=compression)
  
         for grp_id, props in nsh.group_props.items():
             model_grp = pop_grp.create_group('{}'.format(grp_id))
  
             for key, dataset in props.items():
                 try:
                     model_grp.create_dataset(key, data=dataset, compression=compression)
                 except TypeError:  # pragma: no cover
                     str_list = [str(d) for d in dataset]
                     hf.create_dataset(key, data=str_list, compression=compression)


@Timer(name="_write_edge_h5", logger=None) 
def _write_edges_h5(
    merged_edges: MVEdgesCollator,
    pop_name: str,
    n_total_conns: int,
    src_network: str,
    trg_network: str,
    edges_file_name: str,
    compression: str | None
):
    with h5py.File(edges_file_name, 'w') as hf:
        # Initialize the hdf5 groups and datasets
        add_hdf5_attrs(hf)
        pop_grp = hf.create_group('/edges/{}'.format(pop_name))

        pop_grp.create_dataset('source_node_id', (n_total_conns,), dtype='uint64', compression=compression)
        pop_grp['source_node_id'].attrs['node_population'] = src_network
        pop_grp.create_dataset('target_node_id', (n_total_conns,), dtype='uint64', compression=compression)
        pop_grp['target_node_id'].attrs['node_population'] = trg_network
        pop_grp.create_dataset('edge_group_id', (n_total_conns,), dtype='uint16', compression=compression)
        pop_grp.create_dataset('edge_group_index', (n_total_conns,), dtype='uint32', compression=compression)
        pop_grp.create_dataset('edge_type_id', (n_total_conns,), dtype='uint32', compression=compression)

        for group_id in merged_edges.group_ids:
            # different model-groups will have different datasets/properties depending on what edge information
            # is being saved for each edges
            model_grp = pop_grp.create_group(str(group_id))
            for prop_mdata in merged_edges.get_group_metadata(group_id):
                model_grp.create_dataset(prop_mdata['name'], shape=prop_mdata['dim'], dtype=prop_mdata['type'], compression=compression)

        # Uses the collated edges (eg combined edges across all edge-types) to actually write the data to hdf5,
        # potentially in multiple chunks. For small networks doing it this way isn't very effiecent, however
        # this has the benefits:
        #  * For very large networks it won't always be possible to store all the data in memory.
        #  * When using MPI/multi-node the chunks can represent data from different ranks.
        for chunk_id, idx_beg, idx_end in merged_edges.itr_chunks():
            pop_grp['source_node_id'][idx_beg:idx_end] = merged_edges.get_source_node_ids(chunk_id)
            pop_grp['target_node_id'][idx_beg:idx_end] = merged_edges.get_target_node_ids(chunk_id)
            pop_grp['edge_type_id'][idx_beg:idx_end] = merged_edges.get_edge_type_ids(chunk_id)
            pop_grp['edge_group_id'][idx_beg:idx_end] = merged_edges.get_edge_group_ids(chunk_id)
            pop_grp['edge_group_index'][idx_beg:idx_end] = merged_edges.get_edge_group_indices(chunk_id)

            for group_id, prop_name, grp_idx_beg, grp_idx_end in merged_edges.get_group_data(chunk_id):
                prop_array = merged_edges.get_group_property(prop_name, group_id, chunk_id)
                pop_grp[str(group_id)][prop_name][grp_idx_beg:grp_idx_end] = prop_array

from enum import Enum
class MVParMethod(Enum):
    NONE = 0
    ALL_GATHER = 1
    DISTRIBUTED = 2

@t.final
class MVDenseNetwork(MVNetwork):
    def __init__(
        self,
        name:str,
        parallel_method: MVParMethod=MVParMethod.NONE,
        **network_props: t.Any):
        super(MVDenseNetwork, self).__init__(name, **network_props or {})
        # self.__edges_types = {}
        # self.__src_mapping = {}
        # self.__networks = {}
        # self.__node_count = 0
        self.parallel_method : MVParMethod = parallel_method
        self._nodes : list[Node] = []
        self.__edges_tables : list[MVEdgeTypesTable] = []
        self.__all_edges_tables: list[MVEdgeTypesTable] = [] 
        # self._target_networks = {}
        self.__id_map = []
        self.__lookup = []

    @override
    def _initialize(self):
        self.__id_map = []
        self.__lookup = []

    @override
    def _build_ns_nodes(self, node_tuples: list[Node]):
        self._nodes.extend(node_tuples)
        self._nnodes : int = len(self._nodes)

    def edges_table(self):
        return (
            self.__all_edges_tables
            if self.__all_edges_tables
            else self.__edges_tables
        )
    
    @override
    def _save_nodes(self,
                    nodes_file_name: str,
                    mode:str='w',
                    compression:str | None ='gzip'):
        if not self._nodes_built:
            self._build_nodes()
        if compression is None or compression.lower() == 'none':
            compression = None  # legit option for h5py for no compression
        nsh5 = NodesCollator(self._nnodes)
        nsh5.build_from_nodesets(self._node_sets, self.nodes())

        if mpi_rank == 0:
            _write_nodes_h5(nsh5, self.name, nodes_file_name, mode, compression)

        barrier()

    @override
    def nodes_iter(self, nids: Iterable[int]|None=None) -> Iterable[Node]:
        if nids:
            return [n for n in self._nodes if n.node_id in nids]
        else:
            return self._nodes

    # def _process_nodepool(self, nodepool):
    #     return nodepool

    @override
    def import_nodes(
        self,
        nodes_file_name: str,
        node_types_file_name: str,
        population: str|None=None
    ) -> None:
        sonata_file = sonata.File(data_files=nodes_file_name, data_type_files=node_types_file_name)
        if sonata_file.nodes is None:
            raise Exception('nodes file {} does not have any nodes.'.format(nodes_file_name))

        populations = sonata_file.nodes.populations
        if len(populations) == 1:
            node_pop = populations[0]
        elif population is None:
            raise Exception('The nodes file {} contains multiple populations.'.format(nodes_file_name) +
                            'Please specify population parameter.')
        else:
            for pop in populations:
                if pop.name == population:
                    node_pop = pop
                    break
            else:
                raise Exception('Nodes file {} does not contain population {}.'.format(nodes_file_name, population))

        for node_type_props in node_pop.node_types_table:
            self._add_node_type(node_type_props)

        for node in node_pop:
            self._node_id_gen.remove_id(node.node_id)
            self._nodes.append(Node(node.node_id, node.group_props, node.node_type_properties))

    @override
    def _build_edges_pre_process(self):
        pass

    @override
    @Timer(name="M1DenseNetwork__add_edges", logger=None) 
    def _build_cmap_edges(self, connection_map: ConnectionMap, i: int):
        """

        :param connection_map:
        :param i:
        """
        edge_type_id = connection_map.edge_type_properties['edge_type_id']
        logger.debug('Generating edges data for edge_types_id {}.'.format(edge_type_id))
        edges_table = MVEdgeTypesTable(connection_map,
                                       network_name=self.name,
                                       set_props=True)
        edges_table.save()
        logger.debug('Edge-types {} data built with {} connection ({} synapses)'.format(
            edge_type_id, edges_table.n_edges, edges_table.n_syns)
        )
        # TODO: are target networks neeeded?
        # target_net = connection_map.target_nodes
        # self._target_networks[target_net.network_name] = target_net.network
        # To EdgeTypesTable the number of synaptic/gap connections between all source/target paris, which can be more
        # than the number of actual edges stored (for efficency), may be a better user-representation.
        self.__edges_tables.append(edges_table)
        self.increment_edges(edges_table.n_syns)  # edges_table.n_edges

    @override
    def _edge_table_summary(self):
        list_nedges = collect_counts(self._nedges)
        log_at_root(logger, logging.INFO, "NEDGES [%s]", str(list_nedges))

    @override
    def _build_edges_post_process(self):
        self._edge_table_summary()
        if self.parallel_method == MVParMethod.ALL_GATHER:
            self.__all_edges_tables = collect_zipmerge_lists_at_root(
                self.__edges_tables
            )
            self._nedges = accumulate_counts(self._nedges)
            # self._target_networks = dict(
            #     collect_merge_lists(list(self._target_networks.items()))
            # )
        else:
            pass

    @override
    def _save_gap_junctions(self, gj_file_name: str, compression: str|None='gzip', **opts: t.Any):
        source_ids = []
        target_ids = []
        src_gap_ids = []
        trg_gap_ids = []

        if compression == 'none':
            compression = None  # legit option for h5py for no compression

        for et in self.edges_table():
            try:
                is_gap = et.edge_type_properties['is_gap_junction']
            except:
                continue
            if is_gap:
                if et.source_network != et.target_network:
                    raise Exception("All gap junctions must be two cells in the same network builder.")
                table = et.to_dataframe()
                for index, row in table.iterrows():
                    for _ in range(row["nsyns"]): # pyright: ignore[reportArgumentType]
                        source_ids.append(row["source_node_id"])
                        target_ids.append(row["target_node_id"])
                        src_gap_ids.append(self._gj_id_gen.next())
                        trg_gap_ids.append(self._gj_id_gen.next())
            else:
                continue

        if mpi_rank == 0 and len(source_ids) > 0:
            with h5py.File(gj_file_name, 'w') as f:
                add_hdf5_attrs(f)
                f.create_dataset('source_ids', data=np.array(source_ids), compression=compression)
                f.create_dataset('target_ids', data=np.array(target_ids), compression=compression)
                f.create_dataset('src_gap_ids', data=np.array(src_gap_ids), compression=compression)
                f.create_dataset('trg_gap_ids', data=np.array(trg_gap_ids), compression=compression)

    @Timer(name="_collate_edges", logger=None) 
    def _collate_edges(
        self,
        filtered_edge_types: list[MVEdgeTypesTable]
    ):
        seq_flag = self.parallel_method==MVParMethod.ALL_GATHER or mpi_size == 1
        return MVEdgesCollator(
            filtered_edge_types,
            network_name=self.name,
            sequential=seq_flag
        )

    @override
    @Timer(name="_save_edges", logger=None) 
    def _save_edges(
        self,
        edges_file_name: str,
        src_network: str,
        trg_network: str,
        pop_name:str|None=None,
        sort_by: str|None='target_node_id',
        index_by: tuple[t.Any, t.Any]=('target_node_id', 'source_node_id'),
        compression: str|None='gzip',
        sort_on_disk: bool=False,
        **opts: t.Any
    ) -> None:
        barrier()

        if compression == 'none':
            compression = None  # legit option for h5py for no compression

        if mpi_rank == 0:
            logger.debug('Saving {} --> {} edges to {}.'.format(src_network, trg_network, edges_file_name))

        filtered_edge_types = [
            # Some edges may not match the source/target population
            et
            for et in self.edges_table()
            if et.source_network == src_network and et.target_network == trg_network
        ]
        merged_edges = self._collate_edges(filtered_edge_types)
        n_total_conns = merged_edges.n_total_edges
        barrier()


        if n_total_conns == 0:
            if mpi_rank == 0:
                logger.warning(
                    'Was not able to generate any edges using the "connection_rule". Not saving.'
                )
            return

        # Try to sort before writing file, If edges are split across ranks/files for MPI/size issues then we need to
        # write to disk first then sort the hdf5 file
        # sort_on_disk = opts.get('sort_on_disk', False)
        edges_file_name_final = edges_file_name
        if mpi_rank == 0 and sort_by:
            if merged_edges.can_sort and not sort_on_disk:
                merged_edges.sort(sort_by=sort_by)
            else:
                sort_on_disk = True
                edges_file_name_final = edges_file_name

                edges_file_basename = os.path.basename(edges_file_name)
                edges_file_dirname = os.path.dirname(edges_file_name)
                edges_file_name = os.path.join(
                    edges_file_dirname,
                    '.unsorted.{}'.format(edges_file_basename)
                )
                logger.debug(
                        'Unable to sort edges in memory, will temporarly save to {}'.format(edges_file_name) + ' before sorting hdf5 file.')
        #
        pop_name = pop_name if pop_name else f"{src_network}_to_{trg_network}"
        if mpi_rank == 0:
            logger.debug('Saving {} edges to disk'.format(n_total_conns))
            _write_edges_h5(
                merged_edges, pop_name, n_total_conns,
                src_network, trg_network,
                edges_file_name, compression
            )
            if sort_on_disk:
                _sort_on_disk(
                    pop_name, edges_file_name,
                    edges_file_name_final,
                    sort_by,
                    compression)
            if index_by:
                _index_on_disk(index_by, pop_name,
                               edges_file_name_final,
                               compression)
        barrier()
        del merged_edges
        #
        if mpi_rank == 0:
            logger.debug('Saving completed.')

    @override
    def _clear(self):
        self._nedges = 0
        self._nnodes = 0

    @override
    def edges_iter(
        self,
        trg_gids: list[int],
        src_network: str|None=None,
        trg_network: str|None=None
    ) -> Iterable[Edge]:
        matching_edge_tables = self.edges_table()
        if trg_network:
            matching_edge_tables = [et for et in self.edges_table() if et.target_network == trg_network]

        if src_network:
            matching_edge_tables = [et for et in matching_edge_tables if et.source_network == src_network]

        for edge_type_table in matching_edge_tables:
            et_df: pd.DataFrame = edge_type_table.to_dataframe()
            et_df = et_df[et_df['target_node_id'].isin(trg_gids)] #pyright: ignore[reportAssignmentType]
            if len(et_df) == 0:
                continue

            edge_type_props = edge_type_table.edge_type_properties
            for row in et_df.to_dict(orient='records'):
                yield Edge(
                    src_gid=row['source_node_id'],
                    trg_gid=row['target_node_id'],
                    edge_type_props=edge_type_props,
                    syn_props=row
                )
