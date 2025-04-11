import abc
#
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias
from typing_extensions import Self, override
from pydantic import Field
#
from ..base import BaseStruct
from ..util import io as uio, merge_dict_inplace


class DataFile(BaseStruct):
    # path: pathlib.Path
    path : Annotated[str, Field(title="File Path")]

    @override
    def exclude(self) -> set[str]:
        return set([])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        self.path = mod_struct.path
        return self


class DataLink(BaseStruct):
    property_map: Annotated[dict[str, Any], Field(title="Property Map")] = {}

    @override
    def exclude(self) -> set[str]:
        return set([])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        merge_dict_inplace(self.property_map, mod_struct.property_map)
        return self


class ComponentModel(BaseStruct):
    property_map: Annotated[dict[str, Any], Field(title="Property Map")] = {}

    @override
    def exclude(self) -> set[str]:
        return set([])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        merge_dict_inplace(self.property_map, mod_struct.property_map)
        return self


class NeuronModel(BaseStruct):
    N: Annotated[int, Field(title="N")] = 0
    id: Annotated[str, Field(title="ID")] = ""
    proportion: Annotated[float, Field(title="Proportion")] = 0.0
    name: Annotated[str, Field(title="Name")] = ""
    m_type: Annotated[str, Field(title="Model Type")] = ""
    template: Annotated[str, Field(title="Template")] = ""
    dynamics_params: Annotated[str, Field(title="Dynamics Parameters")] = ""
    morphology: Annotated[str, Field(title="Morphology")] = ""
    property_map: Annotated[dict[str, Any], Field(title="Property Map")] = {}
    data_connect: Annotated[list[DataLink], Field(title="Data Connections")] = [] 

    @override
    def exclude(self) -> set[str]:
        return set(["data_connect"])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        if mod_struct.name:
            self.name = mod_struct.name
        # Model Parameters
        if mod_struct.m_type:
            self.m_type = mod_struct.m_type
        if mod_struct.template:
            self.template = mod_struct.template
        if mod_struct.dynamics_params:
            self.dynamics_params = mod_struct.dynamics_params
        # Model Property Map
        for pkey, pvalue in mod_struct.property_map.items():
            if pkey not in self.property_map:
                self.property_map[pkey] = pvalue
            elif pvalue:
                self.property_map[pkey] = pvalue
        return self


class Neuron(BaseStruct):
    ei: Annotated[Literal["e", "i"], Field("E/I")]  # Either e or i
    N: Annotated[int, Field(title="N")] = 0
    fraction: Annotated[float, Field(title="Proportion")] = 0.0
    dims: Annotated[dict[str, Any], Field(title="Dimensions")] = {}
    neuron_models: Annotated[dict[str, NeuronModel], Field(title="Neuron Models")] = {}

    @override
    def exclude(self) -> set[str]:
        return set(["neuron_models"])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        # Fraction and counts
        if mod_struct.fraction > 0:
            self.fraction = mod_struct.fraction
        if mod_struct.N > 0:
            self.N = mod_struct.N # pyright: ignore[reportConstantRedefinition]
        # Neuron dimensions
        for dim_key, dim_value in mod_struct.dims.items():
            if dim_key not in self.dims:
                self.dims[dim_key] = dim_value
            elif dim_value:
                self.dims[dim_key] = dim_value
        # Neuron models
        for mx_name, neuron_mx in mod_struct.neuron_models.items():
            if mx_name not in self.neuron_models:
                self.neuron_models[mx_name] = neuron_mx
            else:
                self.neuron_models[mx_name].apply_mod(neuron_mx)
        return self


class Region(BaseStruct):
    inh_fraction: Annotated[float, Field(title="Inh Fraction")] = 0.0
    region_fraction: Annotated[float, Field(title="Region Fraction")] = 0.0
    ncells: Annotated[int, Field(title="No. Cells")] = 0
    inh_ncells: Annotated[int, Field(title="No. Inh. Cells")] = 0
    exc_ncells: Annotated[int , Field(title="No. Ex. Cells")]= 0
    dims: Annotated[dict[str, Any], Field(title="Dimensions")] = {}
    neurons: Annotated[dict[str, Neuron], Field(title="Neurons")] = {}

    @override
    def exclude(self) -> set[str]:
        return set(["neurons"])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        # update fractions
        if mod_struct.inh_fraction > 0:
            self.inh_fraction = mod_struct.inh_fraction
        if mod_struct.region_fraction > 0:
            self.region_fraction = mod_struct.region_fraction
        # update ncells
        if mod_struct.ncells > 0:
            self.ncells = mod_struct.ncells
        if mod_struct.inh_ncells > 0:
            self.inh_ncells = mod_struct.inh_ncells
        if mod_struct.exc_ncells > 0:
            self.exc_ncells = mod_struct.exc_ncells
        # update dims
        for dkey, dvalue in mod_struct.dims.items():
            if dvalue:
                self.dims[dkey] = dvalue
        # update neuron details
        for nx_name, nx_obj in mod_struct.neurons.items():
            if nx_name not in self.neurons:
                self.neurons[nx_name] = nx_obj
            else:
                self.neurons[nx_name].apply_mod(nx_obj)
        return self

    def find_neuron(self, neuron_name: str) -> Neuron | None:
        if neuron_name in self.neurons:
            return self.neurons[neuron_name]
        return None


class ConnectionModel(BaseStruct):
    target_model_id: Annotated[str, Field(title="Target Id.")] = ""
    source_model_id: Annotated[str, Field(title="Source Id.")] = ""
    weight_max: Annotated[float, Field(title="Max. Weight")] = 0.0
    delay: Annotated[float, Field(title="Delay")] = 0.0
    dynamics_params: Annotated[str, Field(title="Dynamics Parameters")] = ""
    property_map: Annotated[dict[str, Any], Field(title="Property Map")] = {}

    @override
    def exclude(self) -> set[str]:
        return set([])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        # Apply modification
        if mod_struct.target_model_id:
            self.target_model_id = mod_struct.target_model_id
        if mod_struct.source_model_id:
            self.source_model_id = mod_struct.source_model_id
        if mod_struct.delay > 0.0:
            self.delay = mod_struct.delay
        if mod_struct.weight_max > 0.0:
            self.weight_max = mod_struct.weight_max
        # Model Property Map
        for pkey, pvalue in mod_struct.property_map.items():
            if pkey not in self.property_map:
                self.property_map[pkey] = pvalue
            elif pvalue:
                self.property_map[pkey] = pvalue
        return self


class Connection(BaseStruct):
    pre: Annotated[tuple[str, str], Field(title="Pre-Synapse")]
    post: Annotated[tuple[str, str], Field(title="Post-Synapse")]
    probability: Annotated[float, Field(title="Connection Probability")] = 0.0
    connect_models: Annotated[
        dict[str, ConnectionModel], Field(title="Connection Models")
    ] = {}
    property_map: Annotated[dict[str, Any], Field(title="Property Map")] = {}

    @override
    def exclude(self) -> set[str]:
        return set(["connect_models"])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        if mod_struct.pre[0] and mod_struct.pre[1]:
            self.pre = mod_struct.pre
        if mod_struct.post[0] and mod_struct.post[1]:
            self.post = mod_struct.post
        if mod_struct.probability > 0.0:
            self.probability = mod_struct.probability
        # Model Property Map
        for pkey, pvalue in mod_struct.property_map.items():
            if pkey not in self.property_map:
                self.property_map[pkey] = pvalue
            elif pvalue:
                self.property_map[pkey] = pvalue
        # Models
        for mx_name, mx_obj in mod_struct.connect_models.items():
            if mx_name not in self.connect_models:
                self.connect_models[mx_name] = mx_obj
            else:
                self.connect_models[mx_name].apply_mod(mx_obj)
        return self


class ExtNetwork(BaseStruct):
    ncells: int = 0
    locations: dict[str, Region] = {}
    connections: dict[str, Connection] = {}

    @override
    def exclude(self) -> set[str]:
        return set(["locations", "connections"])

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        if mod_struct.ncells > 0:
            self.ncells = mod_struct.ncells
        # Locations
        for c_name, o_cnx in mod_struct.locations.items():
            if c_name not in self.locations:
                self.locations[c_name] = o_cnx
                continue
            self.locations[c_name].apply_mod(o_cnx)
        # Connections
        for c_name, o_cnx in mod_struct.connections.items():
            if c_name not in self.connections:
                self.connections[c_name] = o_cnx
                continue
            self.connections[c_name].apply_mod(o_cnx)
        return self


class Network(BaseStruct):
    ncells: Annotated[int, Field(title = "No. Cells")] = 0
    locations: Annotated[dict[str, Region], Field(title="Regions")] = {}
    connections: Annotated[dict[str, Connection], Field(title="Connections")] = {}
    dims: Annotated[dict[str, Any], Field(title="Dimensions")] = {}
    ext_networks: Annotated[
        dict[str, ExtNetwork], Field(title="Ext. Networks")
    ] = {}
    data_connect: Annotated[list[DataLink], Field(title="Data Connections")] = [] 
    data_files: Annotated[list[DataFile], Field(title="Data Files")] = []

    @override
    def exclude(self) -> set[str]:
        return set(
            [
                "locations",
                "connections",
                "ext_networks",
                "data_connect",
                "data_files"
            ]
        )

    @override
    def apply_mod(self, mod_struct: Self) -> Self:
        # Update dims
        for dkey, dvalue in mod_struct.dims.items():
            if dvalue:
                self.dims[dkey] = dvalue
        # Locations
        for c_name, o_cnx in mod_struct.locations.items():
            if c_name not in self.locations:
                self.locations[c_name] = o_cnx
                continue
            self.locations[c_name].apply_mod(o_cnx)
        # Connections
        for c_name, o_cnx in mod_struct.connections.items():
            if c_name not in self.connections:
                self.connections[c_name] = o_cnx
                continue
            self.connections[c_name].apply_mod(o_cnx)
        # ExtNetwork
        for e_name, e_net in mod_struct.ext_networks.items():
            if e_name not in self.ext_networks:
                self.ext_networks[e_name] = e_net
                continue
            self.ext_networks[e_name].apply_mod(e_net)
        return self

    def populate_ncells(self, N: int) -> "Network":
        """
        Populate cell counts from fractions
        """
        self.ncells = N
        for lx, lrx in self.locations.items():
            ncells_region = int(lrx.region_fraction * N)
            ncells_inh = int(lrx.inh_fraction * ncells_region)
            ncells_exc = ncells_region - ncells_inh
            self.locations[lx].ncells = ncells_region
            self.locations[lx].inh_ncells = ncells_inh
            self.locations[lx].exc_ncells = ncells_exc
            for nx, nurx in lrx.neurons.items():
                eix = nurx.ei
                ncells = ncells_inh if eix == "i" else ncells_exc
                ncells = int(ncells * nurx.fraction)
                if ncells == 0:
                    continue
                self.locations[lx].neurons[nx].N = ncells
        return self

    def find_neuron(self, neuron_name: str) -> Neuron | None:
        for _lx, lrx in self.locations.items():
            neuron_obj = lrx.find_neuron(neuron_name)
            if neuron_obj:
                return neuron_obj
        return None

    @classmethod
    def from_file(cls, in_file: str | Path) -> "Network":
        return cls.model_validate(uio.load(in_file))

    @classmethod
    def from_file_list(cls, in_files: list[str | Path]) -> "Network":
        cust_dict : dict[str, Any] = {}
        for ifile in in_files:
            f_dict = uio.load_json(ifile)
            if cust_dict and f_dict:
                merge_dict_inplace(cust_dict, f_dict)
            else:
                cust_dict = f_dict
        return cls.model_validate(cust_dict)
#
# Mapper Abstract Classes
#
MapperDesc : TypeAlias = dict[str, dict[str, Any]]

class RegionMapper(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name: str, desc: MapperDesc):
        return None

    @abc.abstractmethod
    def neuron_names(self) -> list[str]:
        return []

    @abc.abstractmethod
    def map(self, region_neurons: dict[str, Neuron]) -> Region | None:
        return None


class NeuronMapper(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name: str, desc: MapperDesc):
        return None

    @abc.abstractmethod
    def map(self) -> Neuron | None:
        return None


class ConnectionMapper(abc.ABC):
    @abc.abstractmethod
    def __init__(self, name: str, desc: MapperDesc):
        return None

    @abc.abstractmethod
    def map(self) -> Connection | None:
        return None


class NoneRegionMapper(RegionMapper):
    def __init__(self, name: str, desc: MapperDesc):
        return None

    @override
    def neuron_names(self) -> list[str]:
        return []

    @override
    def map(self, region_neurons: dict[str, Neuron]) -> Region | None:
        return None


class NoneNeuronMapper(NeuronMapper):
    def __init__(self, name: str, desc: MapperDesc):
        return None

    @override
    def map(self) -> Neuron | None:
        return None


class NoneConnectionMapper(ConnectionMapper):
    def __init__(self, name: str, desc: MapperDesc):
        return None

    @override
    def map(self) -> Connection | None:
        return None


def dict2netstruct(
    network_struct: dict[str, Any], # pyright: ignore[reportExplicitt.Any]
) -> Network:
    return Network.model_validate(network_struct)


#
#
def srcdata2network(
    network_desc: dict[str, Any], # pyright: ignore[reportExplicitt.Any]
    model_name: str,
    desc2region_mapper: type[RegionMapper],
    desc2neuron_mapper: type[NeuronMapper],
    desc2connection_mapper: type[ConnectionMapper],
) -> Network:
    loc_struct = {}
    for region, region_desc in network_desc["locations"].items():
        drx_mapper = desc2region_mapper(region, region_desc)
        neuron_struct = {}
        for neuron in drx_mapper.neuron_names():
            if neuron not in region_desc:
                continue
            neuron_desc = region_desc[neuron]
            dn_mapper = desc2neuron_mapper(neuron, neuron_desc)
            neuron_struct[neuron] = dn_mapper.map()
        loc_struct[region] = drx_mapper.map(neuron_struct)
    conn_struct = {}
    for cname, connect_desc in network_desc["connections"].items():
        crx_mapper = desc2connection_mapper(cname, connect_desc)
        conn_struct[cname] = crx_mapper.map()
    return Network(
        name=model_name,
        locations=loc_struct,
        connections=conn_struct,
    )


def subset_network(net_stats: Network, region_list: list[str]) -> Network:
    sub_locs = {k: v for k, v in net_stats.locations.items() if k in region_list}
    return Network(name=net_stats.name, dims=net_stats.dims, locations=sub_locs)


# TODO: Map Node Parameters
# def map_node_paramas(model_struct, node_map):
#     pass
# 
# 
# TODO: Filter node Parameters
# def filter_node_params(model_struct, filter_predicate):
#     pass
# 
# 
# TODO: Map Edge Parameters
# def map_edge_paramas(model_struct, node_map):
#     pass
# 
# 
# TODO: Filter Edge Parameters
# def filter_edge_params(model_struct, filter_predicate):
#     pass
# 
# 
# TODO: Union Parameters
# def union_network(model_struct1, model_struct2):
#     pass
# 
# 
# TODO: Join Network
# def join_network(model_struct1, model_struct2):
#     pass


#
# Example Nework
#
def example_network() -> Network:
    return Network(
        name="v1",
        locations={
            "VISp1": Region(
                name="VISp1",
                inh_fraction=0.4051546391752577,
                region_fraction=0.0696625945317044,
                neurons={
                    "Sst": Neuron(ei="i", fraction=0.0356234096692111),
                    "Lamp5": Neuron(ei="i", fraction=0.7201017811704835),
                    "Sst-Chodl": Neuron(ei="i", fraction=0.0076335877862595),
                    "Pvalb": Neuron(ei="i", fraction=0.0381679389312977),
                    "Vip": Neuron(ei="i", fraction=0.1475826972010178),
                    "GABA-Other": Neuron(ei="i", fraction=0.0508905852417302),
                    "IT": Neuron(ei="e", fraction=0.9896013864818024),
                    "ET": Neuron(ei="e", fraction=0.0),
                    "CT": Neuron(ei="e", fraction=0.0),
                    "NP": Neuron(ei="e", fraction=0.0),
                    "Glut-Other": Neuron(ei="e", fraction=0.0103986135181975),
                },
            ),
            "VISp2/3": Region(
                name="VISp2/3",
                inh_fraction=0.0787551960453881,
                region_fraction=0.3486199987072587,
                neurons={
                    "Sst": Neuron(ei="i", fraction=0.1619115549215406),
                    "Lamp5": Neuron(ei="i", fraction=0.1583452211126961),
                    "Sst-Chodl": Neuron(ei="i", fraction=0.0021398002853067),
                    "Pvalb": Neuron(ei="i", fraction=0.3323823109843081),
                    "Vip": Neuron(ei="i", fraction=0.3166904422253923),
                    "GABA-Other": Neuron(ei="i", fraction=0.028530670470756),
                    "IT": Neuron(ei="e", fraction=1.0),
                    "ET": Neuron(ei="e", fraction=0.0),
                    "CT": Neuron(ei="e", fraction=0.0),
                    "NP": Neuron(ei="e", fraction=0.0),
                    "Glut-Other": Neuron(ei="e", fraction=0.0),
                },
            ),
            "VISp4": Region(
                name="VISp4",
                inh_fraction=0.1119157340355497,
                region_fraction=0.2096987912869239,
                neurons={
                    "Sst": Neuron(ei="i", fraction=0.315126050420168),
                    "Lamp5": Neuron(ei="i", fraction=0.0184873949579831),
                    "Sst-Chodl": Neuron(ei="i", fraction=0.0),
                    "Pvalb": Neuron(ei="i", fraction=0.5327731092436975),
                    "Vip": Neuron(ei="i", fraction=0.1319327731092437),
                    "GABA-Other": Neuron(ei="i", fraction=0.0016806722689075),
                    "IT": Neuron(ei="e", fraction=0.9626178121359736),
                    "ET": Neuron(ei="e", fraction=0.0293338981255956),
                    "CT": Neuron(ei="e", fraction=0.0002117970983797),
                    "NP": Neuron(ei="e", fraction=0.0074128984432913),
                    "Glut-Other": Neuron(ei="e", fraction=0.0004235941967595),
                },
            ),
            "VISp5": Region(
                name="VISp5",
                inh_fraction=0.165427954926876,
                region_fraction=0.1776064895611143,
                neurons={
                    "Sst": Neuron(ei="i", fraction=0.4557971014492754),
                    "Lamp5": Neuron(ei="i", fraction=0.022463768115942),
                    "Sst-Chodl": Neuron(ei="i", fraction=0.0065217391304347),
                    "Pvalb": Neuron(ei="i", fraction=0.4876811594202898),
                    "Vip": Neuron(ei="i", fraction=0.0217391304347826),
                    "GABA-Other": Neuron(ei="i", fraction=0.0057971014492753),
                    "IT": Neuron(ei="e", fraction=0.5212582591209423),
                    "ET": Neuron(ei="e", fraction=0.3230393565067509),
                    "CT": Neuron(ei="e", fraction=0.0439528871014076),
                    "NP": Neuron(ei="e", fraction=0.1041367423154266),
                    "Glut-Other": Neuron(ei="e", fraction=0.0076127549554725),
                },
            ),
            "VISp6a": Region(
                name="VISp6a",
                inh_fraction=0.0627792416084316,
                region_fraction=0.163661043242195,
                neurons={
                    "Sst": Neuron(ei="i", fraction=0.3485401459854014),
                    "Lamp5": Neuron(ei="i", fraction=0.0255474452554744),
                    "Sst-Chodl": Neuron(ei="i", fraction=0.010948905109489),
                    "Pvalb": Neuron(ei="i", fraction=0.5620437956204379),
                    "Vip": Neuron(ei="i", fraction=0.0383211678832116),
                    "GABA-Other": Neuron(ei="i", fraction=0.0145985401459854),
                    "IT": Neuron(ei="e", fraction=0.2159882654932159),
                    "ET": Neuron(ei="e", fraction=0.0006111722283339),
                    "CT": Neuron(ei="e", fraction=0.7593203764820926),
                    "NP": Neuron(ei="e", fraction=0.0053783156093387),
                    "Glut-Other": Neuron(ei="e", fraction=0.0187018701870187),
                },
            ),
            "VISp6b": Region(
                name="VISp6b",
                inh_fraction=0.0529100529100529,
                region_fraction=0.0307510826708034,
                neurons={
                    "Sst": Neuron(ei="i", fraction=0.275),
                    "Lamp5": Neuron(ei="i", fraction=0.0875),
                    "Sst-Chodl": Neuron(ei="i", fraction=0.075),
                    "Pvalb": Neuron(ei="i", fraction=0.475),
                    "Vip": Neuron(ei="i", fraction=0.0375),
                    "GABA-Other": Neuron(ei="i", fraction=0.05),
                    "IT": Neuron(ei="e", fraction=0.0467877094972067),
                    "ET": Neuron(ei="e", fraction=0.0),
                    "CT": Neuron(ei="e", fraction=0.7255586592178771),
                    "NP": Neuron(ei="e", fraction=0.0),
                    "Glut-Other": Neuron(ei="e", fraction=0.2276536312849162),
                },
            ),
        },
    )
