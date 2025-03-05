import logging
import pydantic
from pathlib import Path
import mousev1.model as v1model
import mousev1.operations as v1ops
from airavata_cerebrum.model.setup import RecipeSetup
from airavata_cerebrum.model.recipe import ModelRecipe
from airavata_cerebrum.model.recipe import netstruct_from_file

logging.basicConfig(level=logging.INFO)


class RcpSettings(pydantic.BaseModel):
    name: str = "v1"
    base_dir: Path = Path("./")
    recipe_dir: Path = Path("./v1/recipe/")
    recipe_files: dict[str, list[str | Path]] = {
        "recipe": ["recipe.json"],
        "templates": ["recipe_template.json"]
    }
    custom_mod: Path = Path("./v1/recipe/custom_mod.json")
    ctdb_models_dir: Path = Path("./v1/components/point_neuron_models/")
    nest_models_dir: Path = Path("./v1/components/cell_models/")
    save_flag: bool = True


def model_recipe(cfg_set: RcpSettings):
    md_recipe_setup = RecipeSetup(
        name=cfg_set.name,
        base_dir=cfg_set.base_dir,
        recipe_files=cfg_set.recipe_files,
        recipe_dir=cfg_set.recipe_dir,
        create_model_dir=True,
    )
    custom_mod_struct = netstruct_from_file(cfg_set.custom_mod) 
    return ModelRecipe(
        recipe_setup=md_recipe_setup,
        region_mapper=v1model.V1RegionMapper,
        neuron_mapper=v1model.V1NeuronMapper,
        connection_mapper=v1model.V1ConnectionMapper,
        network_builder=v1model.V1BMTKNetworkBuilder,
        mod_structure=custom_mod_struct,
        save_flag=cfg_set.save_flag,
    )

def struct_bmtk(cfg_set: RcpSettings):
    md_dex = model_recipe(cfg_set)
    md_dex.build_net_struct()
    md_dex.apply_mod()
    md_dex.build_network()

def build_bmtk(cfg_set: RcpSettings):
    md_dex = model_recipe(cfg_set)
    md_dex.download_db_data()
    md_dex.db_post_ops()
    md_dex.map_source_data()
    md_dex.build_net_struct()
    md_dex.apply_mod()
    md_dex.build_network()

def convert_models_to_nest(cfg_set: RcpSettings):
    v1ops.convert_ctdb_models_to_nest(cfg_set.ctdb_models_dir,
                                      cfg_set.nest_models_dir)

if __name__ == "__main__":
    cfg_set = RcpSettings()
    convert_models_to_nest(cfg_set)
    # build_bmtk(cfg_set)
    struct_bmtk(cfg_set)
