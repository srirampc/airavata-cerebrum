import logging
import pydantic
from pathlib import Path
import mousev1.model as v1model
import mousev1.operations as v1ops
from airavata_cerebrum.model.setup import RecipeSetup
from airavata_cerebrum.model.recipe import ModelRecipe
from airavata_cerebrum.model.structure import Network

logging.basicConfig(level=logging.INFO)


class RcpSettings(pydantic.BaseModel):
    name: str = "v1"
    base_dir: Path = Path("./")
    recipe_dir: Path = Path("./v1/recipe/")
    recipe_files: dict[str, list[str | Path]] = {
       "recipe": [
            "recipe.json",
            "recipe_data.json",
            "recipe_dm_l1.json",
            "recipe_dm_l23.json",
            "recipe_dm_l4.json",
            # "recipe_dm_l5.json",
        ],
        "templates": [
            "recipe_template.json"
        ]
    }
    custom_mod: list[str | Path] = [
        Path("./v1/recipe/custom_mod.json"),
        Path("./v1/recipe/custom_mod_l1.json"),
        Path("./v1/recipe/custom_mod_l23.json"),
        Path("./v1/recipe/custom_mod_l4.json"),
        Path("./v1/recipe/custom_mod_ext.json"),
    ]
    ctdb_models_dir: Path = Path("./v1/components/point_neuron_models/")
    nest_models_dir: Path = Path("./v1/components/cell_models/")
    save_flag: bool = True


def recipe_setup(rcp_set: RcpSettings):
    return RecipeSetup(
        name=rcp_set.name,
        base_dir=rcp_set.base_dir,
        recipe_files=rcp_set.recipe_files,
        recipe_dir=rcp_set.recipe_dir,
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
    md_dex.apply_mod()
    md_dex.build_network()

def build_bmtk(rcp_set: RcpSettings):
    md_dex = model_recipe(rcp_set)
    md_dex.download_db_data()
    md_dex.db_post_ops()
    md_dex.map_source_data()
    md_dex.build_net_struct()
    md_dex.apply_mod()
    md_dex.build_network()

def run_nest():
    import nest
    config_file = "./v1/config_nest.json"
    # Instantiate SonataNetwork
    sonata_net = nest.SonataNetwork(config_file)

    # Create and connect nodes
    node_collections = sonata_net.BuildNetwork()
    print("Node Collections", node_collections.keys())
    # Connect spike recorder to a population
    spike_rec = nest.Create("spike_recorder")
    nest.Connect(node_collections["v1"], spike_rec)



def convert_models_to_nest(cfg_set: RcpSettings):
    v1ops.convert_ctdb_models_to_nest(cfg_set.ctdb_models_dir,
                                      cfg_set.nest_models_dir)

if __name__ == "__main__":
    cfg_set = RcpSettings()
    convert_models_to_nest(cfg_set)
    # build_bmtk(cfg_set)
    struct_bmtk(cfg_set)
