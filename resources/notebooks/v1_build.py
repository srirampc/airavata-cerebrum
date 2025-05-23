import argparse
import logging
import pydantic
#
from pathlib import Path
from typing import NamedTuple
#
from nest.lib.hl_api_sonata import SonataNetwork
from nest.lib.hl_api_nodes import Create as NestCreate
from nest.lib.hl_api_connections import Connect as NestConnect
from nest.lib.hl_api_types import NodeCollection
#
import mousev1.model as v1model
import mousev1.operations as v1ops
from airavata_cerebrum.recipe import ModelRecipe, RecipeSetup
from airavata_cerebrum.model.structure import Network
from airavata_cerebrum.util.io import load as loadio

logging.basicConfig(level=logging.INFO)

# Declaring namedtuple()
class NestSonata(NamedTuple):
    net : SonataNetwork | None = None
    spike_rec: NodeCollection | None = None
    multi_meter: NodeCollection | None = None


class RcpSettings(pydantic.BaseModel):
    name: str = "v1"
    ncells: int = 120000
    base_dir: Path = Path("./")
    recipe_dir: Path = Path("./v1/recipe/")
    recipe_output_dir: Path = Path("./v1/recipe/")
    levels: list[str] = ["L1",  "L23", "L4", "L5", "L6"]
    recipe_levels: dict[str, str | Path] = {
        "L1"  : "recipe_dm_l1.json",
        "L23" : "recipe_dm_l23.json",
        "L4"  : "recipe_dm_l4.json",
        "L5"  : "recipe_dm_l5.json",
        "L6"  : "recipe_dm_l6.json",
    }
    custom_mod_main : Path = Path("./v1/recipe/custom_mod.json")
    custom_mod_levels: dict[str, str | Path] = {
        "L1"  : Path("./v1/recipe/custom_mod_l1.json"),
        "L23" : Path("./v1/recipe/custom_mod_l23.json"),
        "L4"  : Path("./v1/recipe/custom_mod_l4.json"),
        "L5"  : Path("./v1/recipe/custom_mod_l5.json"),
        "L6"  : Path("./v1/recipe/custom_mod_l6.json"),
    }
    custom_mod_exts : list[str | Path] = [
        Path("./v1/recipe/custom_mod_ext.json"),
        Path("./v1/recipe/custom_mod_ext_lgn.json"),
        Path("./v1/recipe/custom_mod_ext_bkg.json"),
    ]
    ctdb_models_dir: Path = Path("./v1/components/point_neuron_models/")
    nest_models_dir: Path = Path("./v1/components/cell_models/")
    save_flag: bool = True

    @property
    def recipe_files(self) -> dict[str, list[str | Path]] :
        """The recipe_files property."""
        full_recipe = ["recipe.json", "recipe_data.json"] + [
            self.recipe_levels[lx] for lx in self.levels
        ]
        return {
            "recipe": full_recipe,
            "templates": [ "recipe_template.json" ]
        }

    @property
    def custom_mod(self) -> list[str | Path] :
        return [
            self.custom_mod_main,
        ] + [
            self.custom_mod_levels[lx] for lx in self.levels
        ] + self.custom_mod_exts


def recipe_setup(rcp_set: RcpSettings):
    return RecipeSetup(
        name=rcp_set.name,
        base_dir=rcp_set.base_dir,
        recipe_files=rcp_set.recipe_files,
        recipe_dir=rcp_set.recipe_dir,
        recipe_output_dir=rcp_set.recipe_output_dir,
        create_model_dir=True,
    )


def model_recipe(rcp_set: RcpSettings):
    md_recipe_setup = recipe_setup(rcp_set)
    custom_mod_struct = Network.from_file_list(rcp_set.custom_mod) 
    return ModelRecipe(
        recipe_setup=md_recipe_setup,
        region_mapper=v1model.V1RegionMapper,
        neuron_mapper=v1model.V1NeuronMapper,
        connection_mapper=v1model.V1ConnectionMapper,
        network_builder=v1model.V1BMTKNetworkBuilder,
        mod_structure=custom_mod_struct,
        save_flag=rcp_set.save_flag,
    )


def struct_bmtk(rcp_set: RcpSettings):
    md_dex = model_recipe(rcp_set)
    md_dex.build_net_struct()
    md_dex.apply_mod(rcp_set.ncells)
    md_dex.build_network()


def build_bmtk(rcp_set: RcpSettings):
    md_dex = model_recipe(rcp_set)
    md_dex.download_db_data()
    md_dex.run_db_post_ops()
    md_dex.map_source_data()
    md_dex.build_net_struct()
    md_dex.apply_mod(rcp_set.ncells)
    md_dex.build_network()


def load_nest_sonata(
    nest_config_file: str = "./v1/config_nest.json",
):
    # Instantiate SonataNetwork
    sonata_net = SonataNetwork(nest_config_file)

    # Create and connect nodes
    node_collections = sonata_net.BuildNetwork()
    print("Node Collections", node_collections.keys())

    # Connect spike recorder to a population
    spike_rec = NestCreate("spike_recorder")
    NestConnect(node_collections["v1"], spike_rec)

    # Attach Multimeter
    multi_meter = NestCreate(
        "multimeter",
        params={
            # "interval": 0.05,
            "record_from": [
                "V_m",
                "I",
                "I_syn",
                "threshold",
                "threshold_spike",
                "threshold_voltage",
                "ASCurrents_sum",
            ],
        },
    )
    NestConnect(multi_meter, node_collections["v1"])

    # Simulate the network
    # sonata_net.Simulate()
    return NestSonata(sonata_net, spike_rec, multi_meter)

def convert_models_to_nest(cfg_set: RcpSettings):
    v1ops.convert_ctdb_models_to_nest(
        str(cfg_set.ctdb_models_dir), 
        str(cfg_set.nest_models_dir)
    )

def data_mapped_model(levels: tuple[str,...]=("L1", "L23", "L4")):
    rcp_set = RcpSettings(levels=list(levels))
    mdrcp = model_recipe(rcp_set)
    mdrcp.map_source_data()
    mdrcp.build_net_struct()
    mdrcp.apply_mod(rcp_set.ncells)
    mdrcp.build_network()
    return load_nest_sonata()

def main(input_config_file: str):
    cfg_set = RcpSettings.model_validate(loadio(input_config_file))
    print(cfg_set.model_dump_json(indent=4))
    # convert_models_to_nest(cfg_set)
    # build_bmtk(cfg_set)
    struct_bmtk(cfg_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build V1 SONATA network file."
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="SONATA network configuration.",
    )
    # parser.add_argument(
    #     "-t",
    #     "--threads",
    #     default=8,
    #     type=int,
    #     help="SONATA network configuration.",
    # )
    rargs = parser.parse_args()
    main(rargs.input_file)
 
