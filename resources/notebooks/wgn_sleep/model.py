import typing
from airavata_cerebrum.model import structure
import ast


class WGNRegionMapper(structure.RegionMapper):
    def __init__(self, name: str, region_desc: typing.Dict[str, typing.Dict]):
        self.name = name
        self.region_desc = region_desc
        self.property_map = self.region_desc["property_map"]
        self.src_data = {}


class WGNNeuronMapper(structure.NeuronMapper):
    def __init__(self, name: str, desc: typing.Dict[str, typing.Dict]):
        self.name = name
        self.desc = desc


class WGNConnectionMapper(structure.ConnectionMapper):
    def __init__(self, name: str, desc: typing.Dict[str, typing.Dict]):
        self.name = name
        self.pre, self.post = self.name.split(",")
        self.desc = desc


class WGNNetworkBuilder:
    def __init__(
        self,
        net_struct: structure.Network,
        **kwargs,
    ):
        self.net_struct: structure.Network = net_struct
        self.fraction: float = 1.0
        self.flat: bool = False
 
