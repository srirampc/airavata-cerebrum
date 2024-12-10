import pathlib
import typing
import os
import logging
import pydantic

from ..util import io as cbmio


def _log():
    return logging.getLogger(__name__)


#
# Lookup Keys in Configuration dictionary
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
class RecipeLabels:
    INIT_PARAMS = " Init Arguments : "
    EXEC_PARAMS = " Exec Arguments : "
    NA = " N/A "
 
#
# Class for structure of Recipes
class RecipeSetup(pydantic.BaseModel):
    recipe_dir: str | pathlib.Path = pathlib.Path(".")
    recipe_sections: typing.Dict[str, typing.Any] = {}
    recipe_files: typing.Dict[str, typing.List[str | pathlib.Path]]
    #
    create_model_dir: bool = False
    name: str
    base_dir: str | pathlib.Path

    @property
    def model_dir(self) -> pathlib.Path:
        return pathlib.Path(self.base_dir, self.name)

    @property
    def network_dir(self) -> pathlib.Path:
        return pathlib.Path(self.base_dir, self.name, RecipeKeys.NETWORK)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Check if recipe available
        if RecipeKeys.RECIPE in self.recipe_files:
            self.load_sections()
        else:
            try:
                self.load_section_list()
            except KeyError:
                _log().error(
                    "Failed to find one of two Recipe Keys: [%s]", str(RecipeKeys.RECIPE_SECTIONS)
                )
        # Create model directory
        if self.create_model_dir and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # Recipe View Templates
        if RecipeKeys.TEMPLATES in self.recipe_files:
            _log().warning("Loading View Templates")
            for cfg_file in self.recipe_files[RecipeKeys.TEMPLATES]:
                cfg_dict = cbmio.load(self.recipe_file_path(cfg_file))
                if cfg_dict:
                    self.update_section(cfg_dict, RecipeKeys.TEMPLATES)

    def recipe_file_path(self, file_name: str | pathlib.Path) -> str | pathlib.Path:
        if self.recipe_dir:
            return pathlib.Path(self.recipe_dir, file_name)
        return file_name

    def recipe_output_prefix(self, cfg_key: str) -> str:
        return cfg_key + "_output"

    def get_section(self, cfg_key: str) -> typing.Dict[str, typing.Any]:
        return self.recipe_sections[cfg_key]

    def update_section(
        self,
        section_dict: typing.Dict[str, typing.Any],
        section_key: str | None = None,
    ) -> None:
        if section_key:
            if section_key in self.recipe_sections:
                self.recipe_sections[section_key] |= section_dict
            else:
                self.recipe_sections[section_key] = section_dict
        else:
            self.recipe_sections |= section_dict

    def load_sections(self) -> None:
        for rcp_file in self.recipe_files[RecipeKeys.RECIPE]:
            rcp_dict = cbmio.load(self.recipe_file_path(rcp_file))
            if rcp_dict:
                self.update_section(rcp_dict)

    def load_section_list(
        self,
    ) -> None:
        for rcp_key in RecipeKeys.RECIPE_SECTIONS:
            for rcp_file in self.recipe_files[rcp_key]:
                rcp_dict = cbmio.load(self.recipe_file_path(rcp_file))
                if rcp_dict:
                    self.update_section(rcp_dict, rcp_key)

    def valid(self) -> bool:
        return all(cfg_key in self.recipe_sections for cfg_key in RecipeKeys.RECIPE_SECTIONS)

    def get_template_for(self, reg_key: str) -> typing.Dict[str, typing.Any]:
        return self.recipe_sections[RecipeKeys.TEMPLATES][reg_key]

    def get_templates(self) -> typing.Dict[str, typing.Any]:
        return self.recipe_sections[RecipeKeys.TEMPLATES]


def init_model_setup(
    name,
    model_base_dir,
    recipe_files,
    recipe_dir,
) -> RecipeSetup:
    return RecipeSetup(
            name=name,
            base_dir=model_base_dir,
            recipe_files=recipe_files,
            recipe_dir=recipe_dir,
    )
