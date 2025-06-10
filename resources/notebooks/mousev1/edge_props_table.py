import typing as t
import numpy as np
import numpy.typing as npt
import hashlib
import pandas as pd

from bmtk.builder.connection_map import ConnectionMap
from bmtk.builder.node import Node

from codetiming import Timer

NPIntArray = npt.NDArray[np.integer[t.Any]]

class MVEdgeTypesTable(object):
    """A class for creating and storing the actual connectivity matrix plus all the possible (hdf5 bound) properties
    of an edge - unlike the ConnectionMap class which only stores the unevaluated rules each edge-types. There should
    be one EdgeTypesTable for each call to Network.add_edges(...)

    By default edges in the SONATA edges.h5 table are stored in a (sparse) SxT table, S/T the num of source/target
    cells. If individual edge properties (syn_weight, syn_location, etc) and added then it must be stored in a SxTxN
    table, N the avg. number of synapses between each source/target pair. The actually number of edges (ie rows)
    saved in the SONATA file will vary.
    """
    def __init__(
        self,
        conn_map: ConnectionMap,
        network_name: str,
        set_props: bool = False,
        set_node_maps: bool = True,
    ):
        # self._connection_map : ConnectionMap = connection_map
        self._network_name : str = network_name
        self.source_network : str = conn_map.source_nodes.network_name
        self.target_network : str = conn_map.target_nodes.network_name
        self.edge_type_id : int = conn_map.edge_type_properties['edge_type_id']
        self.edge_group_id : int = -1  # Will be assigned later during save_edges
        # Create the nsyns table to store the num of synapses/edges between
        # each possible source/target node pair
        self._nsyns_idx2src: list[int] = (
            [n.node_id for n in conn_map.source_nodes]
            if conn_map.source_nodes else []
        )
        self._nsyns_src2idx: dict[int, int] = {
            node_id: i for i, node_id in enumerate(self._nsyns_idx2src)
        }
        self._nsyns_idx2trg: list[int] = (
            [n.node_id for n in conn_map.target_nodes]
            if conn_map.target_nodes else []
        )
        self._nsyns_trg2idx: dict[int, int] = {
            node_id: i for i, node_id in enumerate(self._nsyns_idx2trg)
        }
        self._nsyns_updated : bool= False
        self._n_syns : int = 0
        self.nsyn_table: NPIntArray = np.zeros(
            (len(self._nsyns_idx2src), len(self._nsyns_idx2trg)),
            dtype=np.uint32
        )
        self._edge_type_properties: dict[str, t.Any] = conn_map.edge_type_properties
        #
        self._prop_vals: dict[str, t.Any] = {}  # used to store the arrays for each property
        self._prop_node_ids : NPIntArray | None = None  # used to save the source_node_id and target_node_id for each edge
        #
        self._source_nodes_map: dict[int, Node]= {} # map source_node_id --> Node object
        self._target_nodes_map : dict[int, Node]= {} # map target_node_id --> Node object
        self._property_hash: str = ""
        if set_node_maps:
            self.__init_node_maps(conn_map)
        if set_props: 
            self.__set_nsys(conn_map)
            self.__set_prop_data(conn_map)
            self.__set_property_hash()

    def __set_property_hash(self):
        """Creates a hash key for edge-types based on their (hdf5) properties, for grouping together properties of
        different edge-types. If two edge-types have the same (hdf5) properties they should have the same hash value.
        """
        prop_keys = ['{}({})'.format(p['name'], p['dtype']) for p in self.get_property_metatadata()]
        prop_keys.sort()
        #
        # NOTE: python's hash() function is randomized which is a problem
        # when using MPI to process different edge types across different ranks.
        prop_keys = ':'.join(prop_keys).encode('utf-8')
        self._property_hash = hashlib.md5(prop_keys).hexdigest()[:9]

    def __init_node_maps(self, connection_map: ConnectionMap):
        self._source_nodes_map = { # map source_node_id --> Node object
            s.node_id: s for s in connection_map.source_nodes
        } if connection_map.source_nodes else {}
        self._target_nodes_map = { # map target_node_id --> Node object
            t.node_id: t for t in connection_map.target_nodes
        } if connection_map.target_nodes else {}

    @property
    def n_syns(self):
        """Number of synapses."""
        if self._nsyns_updated:
            self._nsyns_updated = False
            self._n_syns = int(np.sum(self.nsyn_table))
        return self._n_syns

    @property
    def n_edges(self):
        """Number of unque edges/connections (eg rows in SONATA edges file). When multiple synapse can be safely
        represented with just one edge it will have n_edges < n_syns.
        """
        if self._prop_vals:
            return self.n_syns
        else:
            return np.count_nonzero(self.nsyn_table)

    @property
    def edge_type_node_ids(self):
        """Returns a table n_edges x 2, first column containing source_node_ids and second target_node_ids."""
        if self._prop_node_ids is None or self._nsyns_updated:
            if len(self._prop_vals) == 0:
                # Get the source and target node ids from the rows/columns of nsyns table cells that are greater than 0
                nsyn_table_flat = self.nsyn_table.ravel()
                src_trg_prods = np.array(np.meshgrid(self._nsyns_idx2src, self._nsyns_idx2trg)).T.reshape(-1, 2)
                nonzero_idxs = np.argwhere(nsyn_table_flat > 0).flatten()
                self._prop_node_ids = src_trg_prods[nonzero_idxs, :].astype(np.uint64)

            else:
                # If there are synaptic properties go through each source/target pair and add their node-ids N times,
                # where N is the number of synapses between the two nodes
                self._prop_node_ids = np.zeros((self.n_edges, 2), dtype=np.int64)
                idx = 0
                for r, src_id in enumerate(self._nsyns_idx2src):
                    for c, trg_id in enumerate(self._nsyns_idx2trg):
                        nsyns = self.nsyn_table[r, c]

                        self._prop_node_ids[idx:(idx + nsyns), 0] = src_id
                        self._prop_node_ids[idx:(idx + nsyns), 1] = trg_id
                        idx += nsyns

        return self._prop_node_ids

    @Timer(name="MVEdgeTypesTable__set_nsys", logger=None) 
    def __set_nsys(self, connection_map : ConnectionMap):
        valid_conn_itr = (
            (s, t, ns)
            for s,t,ns in connection_map.connection_itr() if ns
        )
        stn_array: npt.NDArray[t.Any] = np.fromiter(
            valid_conn_itr,
            dtype=np.dtype((int, 3))
        )
        self.set_nsyns_stn(stn_array)
        #
        # iterate through all possible SxT source/target pairs and use the user-defined function/list/value to update
        # the number of syns between each pair.
        # TODO: See if this can be vectorized easily.
        # for conn in connections:
        #     if conn[2]:
        #         edges_table.set_nsyns(source_id=conn[0], target_id=conn[1], nsyns=conn[2])
        return self

    @Timer(name="MVEdgeTypesTable__set_params", logger=None) 
    def __set_prop_data(self, connection_map : ConnectionMap):
        # For when the user specified individual edge properties to be put in the hdf5 (syn_weight, syn_location, etc),
        # get prop value and add it to the edge-types table. Need to fetch and store SxTxN value (where N is the avg
        # num of nsyns between each source/target pair) and it is necessary that the nsyns table be finished.
        for param in connection_map.params:
            rule = param.rule
            rets_multiple_vals = isinstance(param.names, (list, tuple, np.ndarray))

            if not rets_multiple_vals:
                prop_name = param.names  # name of property
                prop_type = param.dtypes.get(prop_name, None)
                self.create_property(prop_name=param.names, prop_type=prop_type)  # initialize property array

                for source_node, target_node, edge_index in self.iter_edges():
                    # calls connection map rule and saves value to edge table
                    pval = rule(source_node, target_node)
                    self.set_property_value(prop_name=prop_name, edge_index=edge_index, prop_value=pval)

            else:
                # Same as loop above, but some connection-map 'rules' will return multiple properties for each edge.
                pnames = param.names
                ptypes = [param.dtypes[pn] for pn in pnames]
                for prop_name, prop_type in zip(pnames, ptypes):
                    self.create_property(prop_name=prop_name, prop_type=prop_type)  # initialize property arrays

                for source_node, target_node, edge_index in self.iter_edges():
                    pvals = rule(source_node, target_node)
                    for pname, pval in zip(pnames, pvals):
                        self.set_property_value(prop_name=pname, edge_index=edge_index, prop_value=pval)


    @property
    def source_nodes_map(self):
        return self._source_nodes_map

    @property
    def target_nodes_map(self):
        return self._target_nodes_map

    @property
    def hash_key(self):
        if not self._property_hash:
            self.__set_property_hash()
        return self._property_hash

    @property
    def edge_type_properties(self) -> dict[str, t.Any]:
        return self._edge_type_properties

    def get_property_metatadata(self) -> list[dict[str, str | np.dtype[t.Any]]]:
        if not self._prop_vals:
            return [{'name': 'nsyns', 'dtype': self.nsyn_table.dtype}]
        else:
            return [{'name': pname, 'dtype': pvals.dtype}
                    for pname, pvals in self._prop_vals.items()]

    def set_nsyns(self, source_id: int, target_id: int, nsyns: int):
        assert(nsyns >= 0)
        indexed_pair = (self._nsyns_src2idx[source_id], self._nsyns_trg2idx[target_id])
        self.nsyn_table[indexed_pair] = nsyns
        self._nsyns_updated = True

    def set_nsyns_stn(self, stn_array: NPIntArray):
        """
        Setting number of synapses to nedges

        stn_array: array of size nedges x 3, where
            - first column: source node
            - second column: target node
            - third column: number of synapse
        """
        assert(stn_array is not None)
        self.nsyn_table[
            np.fromiter((self._nsyns_src2idx[s] for s in stn_array[:, 0]),
                        dtype=int,
                        count=stn_array.shape[0]),
            np.fromiter((self._nsyns_trg2idx[t] for t in stn_array[:, 1]),
                        dtype=int,
                        count=stn_array.shape[0])
        ] = stn_array[:, 2]
        self._nsyns_updated = True
    
    def create_property(
        self,
        prop_name: str,
        prop_type: np.dtype[t.Any] | None=None
    ):
        assert(prop_name not in self._prop_vals)
        prop_size = self.n_syns
        self._prop_vals[prop_name] = np.zeros(prop_size, dtype=prop_type)

    def iter_edges(self):
        prop_node_ids = self.edge_type_node_ids
        src_nodes_lu = self.source_nodes_map
        trg_nodes_lu = self.target_nodes_map
        for edge_index in range(self.n_edges):
            src_id = prop_node_ids[edge_index, 0]
            trg_id = prop_node_ids[edge_index, 1]
            source_node = src_nodes_lu[src_id]
            target_node = trg_nodes_lu[trg_id]

            yield source_node, target_node, edge_index

    def set_property_value(
        self,
        prop_name: str,
        edge_index: int,
        prop_value: t.Any
    ):
        self._prop_vals[prop_name][edge_index] = prop_value

    def get_property_value(self, prop_name: str) -> t.Any:
        if prop_name == 'nsyns':
            nsyns_table_flat = self.nsyn_table.ravel()
            nonzero_indxs = np.argwhere(nsyns_table_flat > 0).flatten()
            #
            return nsyns_table_flat[nonzero_indxs]
        else:
            return self._prop_vals[prop_name]

    def to_dataframe(self, **kwargs: t.Any):  #pyright: ignore[reportUnusedParameter]
        src_trg_ids = self.edge_type_node_ids
        ret_df = pd.DataFrame({
            'source_node_id': src_trg_ids[:, 0],
            'target_node_id': src_trg_ids[:, 1],
            # 'edge_type_id': self.edge_type_id
        })
        for edge_prop in self.get_property_metatadata():
            pname: str = str(edge_prop['name'])
            ret_df[pname] = self.get_property_value(prop_name=pname)
        #
        return ret_df

    def save(self):
        pass

    def free_data(self):
        del self.nsyn_table
        del self._prop_vals
