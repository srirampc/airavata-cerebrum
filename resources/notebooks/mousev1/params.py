import logging
from pathlib import Path

import pydantic

from airavata_cerebrum.model.structure import Network
from airavata_cerebrum.recipe import ModelRecipe, RecipeSetup

from .model import (V1BMTKNetworkBuilder, V1ConnectionMapper, V1NeuronMapper,
                    V1RegionMapper)


class RecipeParams(pydantic.BaseModel):
    name: str = "v1"
    ncells: int = 30000
    enable_timing: bool = True
    logging_level: int = logging.DEBUG
    root_logging_level: int = logging.DEBUG
    base_dir: Path = Path("./model_builds/v1030")
    recipe_dir: Path = Path("./model_recipes/v1")
    recipe_output_dir: Path | None = Path("./model_data/v1")
    levels: list[str] = ["L1", "L23", "L4", "L5", "L6"]
    recipe_levels: dict[str, str | Path] = {
        "L1": "recipe_dm_l1.json",
        "L23": "recipe_dm_l23.json",
        "L4": "recipe_dm_l4.json",
        "L5": "recipe_dm_l5.json",
        "L6": "recipe_dm_l6.json",
    }
    custom_mod_main: Path = Path("./model_recipes/v1/custom_mod.json")
    custom_mod_levels: dict[str, str | Path] = {
        "L1": Path("./model_recipes/v1/custom_mod_l1.json"),
        "L23": Path("./model_recipes/v1/custom_mod_l23.json"),
        "L4": Path("./model_recipes/v1/custom_mod_l4.json"),
        "L5": Path("./model_recipes/v1/custom_mod_l5.json"),
        "L6": Path("./model_recipes/v1/custom_mod_l6.json"),
    }
    custom_mod_exts: list[str | Path] = [
        Path("./model_recipes/v1/custom_mod_ext.json"),
        Path("./model_recipes/v1/custom_mod_ext_lgn.json"),
        Path("./model_recipes/v1/custom_mod_ext_bkg.json"),
    ]
    ctdb_models_dir: Path = Path("./model_components/point_neuron_models/")
    nest_models_dir: Path = Path("./model_components/cell_models/")
    save_flag: bool = True

    @property
    def output_dir(self) -> Path:
        return Path(
            self.recipe_output_dir if self.recipe_output_dir else self.recipe_dir
        )

    @property
    def recipe_files(self) -> dict[str, list[str | Path]]:
        """The recipe_files property."""
        full_recipe = ["recipe.json", "recipe_data.json"] + [
            self.recipe_levels[lx] for lx in self.levels
        ]
        return {"recipe": full_recipe, "templates": ["recipe_template.json"]}

    @property
    def custom_mod(self) -> list[str | Path]:
        return (
            [
                self.custom_mod_main,
            ]
            + [self.custom_mod_levels[lx] for lx in self.levels]
            + self.custom_mod_exts
        )

    def recipe_setup(self):
        return RecipeSetup(
            name=self.name,
            base_dir=self.base_dir,
            recipe_files=self.recipe_files,
            recipe_dir=self.recipe_dir,
            recipe_output_dir=self.output_dir,
            create_model_dir=True,
        )

    def model_recipe(self) -> ModelRecipe:
        md_recipe_setup = self.recipe_setup()
        custom_mod_struct = Network.from_file_list(self.custom_mod)
        return ModelRecipe(
            recipe_setup=md_recipe_setup,
            region_mapper=V1RegionMapper,
            neuron_mapper=V1NeuronMapper,
            connection_mapper=V1ConnectionMapper,
            network_builder=V1BMTKNetworkBuilder,
            mod_structure=custom_mod_struct,
            save_flag=self.save_flag,
        )
