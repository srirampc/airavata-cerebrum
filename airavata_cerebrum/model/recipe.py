import logging
import pathlib
import typing
import os

import pydantic

from ..util import io as cbmio
from .. import workflow
from .setup import RecipeSetup, RecipeKeys
from . import structure


def _log():
    return logging.getLogger(__name__)


# File paths
class RecipePaths:
    RECIPE_IO_DIR = "recipe"


class ModelRecipe(pydantic.BaseModel):
    recipe_setup: RecipeSetup
    region_mapper: typing.Type[structure.RegionMapper]
    neuron_mapper: typing.Type[structure.NeuronMapper]
    connection_mapper: typing.Type[structure.ConnectionMapper]
    network_builder: typing.Type
    mod_structure: structure.Network | None = None
    save_flag: bool = True
    out_format: typing.Literal["json", "yaml", "yml"] = "json"
    network_struct: structure.Network = structure.Network(name="empty")

    def output_location(self, recipe_key: str) -> pathlib.Path:
        file_name = self.recipe_setup.recipe_output_prefix(recipe_key)
        out_path = pathlib.Path(
            self.recipe_setup.model_dir, RecipePaths.RECIPE_IO_DIR, file_name
        )
        if not os.path.exists(out_path.parent):
            os.makedirs(out_path.parent)
        return out_path.with_suffix("." + self.out_format)

    def download_db_data(self) -> typing.Dict[str, typing.Any]:
        db_src_config = self.recipe_setup.get_section(RecipeKeys.SRC_DATA)
        _log().info("Start Query and Download Data")
        db_connect_output = workflow.run_db_connect_workflows(db_src_config)
        if self.save_flag:
            cbmio.dump(
                db_connect_output,
                self.output_location(RecipeKeys.DB_CONNECT),
                indent=4,
            )
        _log().info("Completed Query and Download Data")
        return db_connect_output

    def db_post_ops(self):
        db_connect_key = RecipeKeys.DB_CONNECT
        db_datasrc_key = RecipeKeys.SRC_DATA
        db_connect_data = cbmio.load(self.output_location(db_connect_key))
        db_src_config = self.recipe_setup.get_section(RecipeKeys.SRC_DATA)
        db_post_op_data = None
        if db_connect_data:
            db_post_op_data = workflow.run_ops_workflows(
                db_connect_data, db_src_config, RecipeKeys.POST_OPS
            )
        if self.save_flag and db_post_op_data:
            cbmio.dump(db_post_op_data, self.output_location(db_datasrc_key), indent=4)
        return db_post_op_data

    def map_source_data(self):
        db2model_map = self.recipe_setup.get_section(RecipeKeys.DB2MODEL_MAP)
        db_lox_map = db2model_map[RecipeKeys.LOCATIONS]
        db_conn_map = db2model_map[RecipeKeys.CONNECTIONS]
        db_source_data = cbmio.load(self.output_location(RecipeKeys.SRC_DATA))
        srcdata_map_output = None
        if db_source_data:
            db2location_output = workflow.map_srcdata_locations(
                db_source_data, db_lox_map
            )
            db2connect_output = workflow.map_srcdata_connections(
                db_source_data, db_conn_map
            )
            srcdata_map_output = {
                "locations": db2location_output,
                "connections": db2connect_output,
            }
        if self.save_flag and srcdata_map_output:
            cbmio.dump(
                srcdata_map_output,
                self.output_location(RecipeKeys.DB2MODEL_MAP),
                indent=4,
            )
        return srcdata_map_output

    def build_net_struct(self):
        network_desc_output = cbmio.load(self.output_location(RecipeKeys.DB2MODEL_MAP))
        if not network_desc_output:
            return None
        self.network_struct = structure.srcdata2network(
            network_desc_output,
            self.recipe_setup.name,
            self.region_mapper,
            self.neuron_mapper,
            self.connection_mapper,
        )
        return self.network_struct

    def apply_mod(self):
        if self.mod_structure:
            # Update user preference
            self.network_struct = self.network_struct.apply_mod(self.mod_structure)
        # Estimate NCells from the fractions
        self.network_struct.populate_ncells(30000)
        if self.save_flag and self.network_struct:
            cbmio.dump(
                self.network_struct.model_dump(),
                self.output_location(RecipeKeys.NETWORK_STRUCT),
                indent=4,
            )
        return self.network_struct

    def build_bmtk(self):
        # Construct model
        net_builder = self.network_builder(self.network_struct)
        bmtk_net = net_builder.build()
        bmtk_net.save(str(self.recipe_setup.network_dir))
        net_builder.bkg_net.save(str(self.recipe_setup.network_dir))
        return net_builder

def netstruct_from_file(struct_file: pathlib.Path) -> structure.Network | None:
    if struct_file:
        network_struct = cbmio.load(struct_file)
        if network_struct:
            return structure.dict2netstruct(network_struct)
    return None
