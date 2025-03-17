import logging
import os
import typing as t
import pydantic

from pathlib import Path

from ..util import io as cbmio, merge_dict_inplace


def _log():
    return logging.getLogger(__name__)


#
# Lookup Keys in Configuration dictionary
@t.final
class RecipeKeys:
    RECIPE = "recipe"
    # Recipe Sections
    POST_OPS = "post_ops"
    DB2MODEL_MAP = "data2model_map"
    SRC_DATA = "source_data"
    TEMPLATES = "templates"
    #
    NODE_KEY = "node_key"
    NETWORK_STRUCT = "network_structure"
    NETWORK = "network"
    #
    LOCATIONS = "locations"
    CONNECTIONS = "connections"
    INIT_PARAMS = "init_params"
    EXEC_PARAMS = "exec_params"
    LABEL = "label"
    NAME = "name"
    TYPE = "type"
    WORKFLOW = "workflow"
    DB_CONNECT = "db_connect"
    #
    RECIPE_SECTIONS = [DB2MODEL_MAP, SRC_DATA]
#
# Lookup Keys in Configuration dictionary
@t.final
class RecipeLabels:
    INIT_PARAMS = " Init Arguments : "
    EXEC_PARAMS = " Exec Arguments : "
    NA = " N/A "
 
#
# Class for structure of Recipes
class RecipeSetup(pydantic.BaseModel):
    recipe_dir: str | Path = Path(".")
    recipe_sections: dict[str, t.Any] = {}
    recipe_templates: dict[str, t.Any] = {}
    recipe_files: dict[str, list[str | Path]]
    #
    create_model_dir: bool = False
    name: str
    base_dir: str | Path

    @property
    def model_dir(self) -> Path:
        return Path(self.base_dir, self.name)

    @property
    def network_dir(self) -> Path:
        return Path(self.base_dir, self.name, RecipeKeys.NETWORK)

    def __init__(self, **kwargs: t.Any):
        super().__init__(**kwargs)
        self.load_recipe()
        # Recipe View Templates
        self.load_templates()
        # Create model directory
        if self.create_model_dir and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def load_recipe(self):
        # Load if recipe available, load sections otherwise
        if RecipeKeys.RECIPE in self.recipe_files:
            self.load_sections()
        else:
            try:
                self.load_section_list()
            except KeyError:
                _log().error(
                    "Failed to find one of two Recipe Keys: [%s]",
                    str(RecipeKeys.RECIPE_SECTIONS)
                )

    def load_templates(self):
        if RecipeKeys.TEMPLATES in self.recipe_files:
            _log().warning("Loading View Templates")
            for t_file in self.recipe_files[RecipeKeys.TEMPLATES]:
                t_dict = cbmio.load(self.recipe_file_path(t_file))
                if t_dict:
                    merge_dict_inplace(
                        self.recipe_templates,
                        t_dict
                    )

    def recipe_file_path(self, file_name: str | Path) -> str | Path:
        if self.recipe_dir:
            return Path(self.recipe_dir, file_name)
        return file_name

    def recipe_output_prefix(self, cfg_key: str) -> str:
        return cfg_key + "_output"

    def get_section(self, cfg_key: str) -> dict[str, t.Any]:
        return self.recipe_sections[cfg_key]

    def load_sections(self) -> None:
        for rcp_file in self.recipe_files[RecipeKeys.RECIPE]:
            rcp_dict = cbmio.load(self.recipe_file_path(rcp_file))
            if rcp_dict and self.recipe_sections:
                merge_dict_inplace(
                    self.recipe_sections,
                    rcp_dict
                )
            elif rcp_dict:
                self.recipe_sections = rcp_dict

    def load_section_list(
        self,
    ) -> None:
        for rcp_key in RecipeKeys.RECIPE_SECTIONS:
            for rcp_file in self.recipe_files[rcp_key]:
                rcp_dict = cbmio.load(self.recipe_file_path(rcp_file))
                if rcp_dict and self.recipe_sections:
                    merge_dict_inplace(
                        self.recipe_sections,
                        {rcp_key: rcp_dict}
                    )  
                elif rcp_dict:
                    self.recipe_sections[rcp_key] = rcp_dict

    def valid(self) -> bool:
        return all(
            (
                cfg_key in self.recipe_sections
                for cfg_key in RecipeKeys.RECIPE_SECTIONS
            )
        )

    def get_template_for(self, reg_key: str) -> dict[str, t.Any]:
        return self.recipe_templates[reg_key]

    def get_templates(self) -> dict[str, t.Any]:
        return self.recipe_templates


def init_model_setup(
    name: str,
    model_base_dir: str| Path,
    recipe_files: dict[str, list[str | Path]],
    recipe_dir: str| Path,
) -> RecipeSetup:
    return RecipeSetup(
            name=name,
            base_dir=model_base_dir,
            recipe_files=recipe_files,
            recipe_dir=recipe_dir,
    )
