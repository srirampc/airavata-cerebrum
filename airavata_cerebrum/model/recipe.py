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
@typing.final
class RecipePaths:
    RECIPE_IO_DIR = "recipe"


class ModelRecipe(pydantic.BaseModel):
    recipe_setup: RecipeSetup
    region_mapper: type[structure.RegionMapper]
    neuron_mapper: type[structure.NeuronMapper]
    connection_mapper: type[structure.ConnectionMapper]
    network_builder: type
    mod_structure: structure.Network | None = None
    save_flag: bool = True
    write_duck: bool = True
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

    def save_db_out(
        self,
        db_connect_output: dict[str, typing.Any] | None ,
        db_out_loc: str | pathlib.Path,
        write_duck: bool,
    ) -> None:
        if db_connect_output is None: 
            return
        if self.save_flag:
            cbmio.dump(
                db_connect_output,
                db_out_loc,
                indent=4,
            )
        if write_duck:
            duck_out_loc = str(db_out_loc).replace(self.out_format, 'db')
            workflow.write_db_connect_duck(
                db_connect_output,
                duck_out_loc,
            )

    def download_db_data(self) -> dict[str, typing.Any]:
        db_src_config = self.recipe_setup.get_section(RecipeKeys.SRC_DATA)
        _log().info("Start Query and Download Data")
        db_connect_output = workflow.run_db_connect_workflows(db_src_config)
        #
        db_out_loc = self.output_location(RecipeKeys.DB_CONNECT)
        self.save_db_out(db_connect_output, db_out_loc, self.write_duck)
        _log().info("Completed Query and Download Data")
        return db_connect_output

    def run_db_post_ops(self):
        db_connect_data = cbmio.load(
            self.output_location(RecipeKeys.DB_CONNECT)
        )
        db_src_config = self.recipe_setup.get_section(RecipeKeys.SRC_DATA)
        db_post_op_data = None
        _log().info("Start DB Post ops")
        if db_connect_data:
            db_post_op_data = workflow.run_ops_workflows(
                db_connect_data,
                db_src_config,
                RecipeKeys.POST_OPS
            )
            #
            db_out_loc = self.output_location(RecipeKeys.SRC_DATA)
            self.save_db_out(db_post_op_data, db_out_loc, False)
        _log().info("Completed Post ops")
        return db_post_op_data

    def acquire_source_data(self):
        db_connection_output = self.download_db_data()
        db_post_op_data = self.run_db_post_ops()
        return db_connection_output, db_post_op_data

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
        network_desc_output = cbmio.load(
            self.output_location(RecipeKeys.DB2MODEL_MAP)
        )
        if not network_desc_output:
            network_desc_output = self.map_source_data()
        if network_desc_output:
            self.network_struct = structure.srcdata2network(
                network_desc_output,
                self.recipe_setup.name,
                self.region_mapper,
                self.neuron_mapper,
                self.connection_mapper,
            )
            return self.network_struct

    def source_data2model_struct(self):
        self.map_source_data()
        return self.build_net_struct()

    def apply_mod(self, ncells: int=30000):
        if self.mod_structure:
            # Update user preference
            self.network_struct = self.network_struct.apply_mod(self.mod_structure)
        # Estimate NCells from the fractions
        self.network_struct.populate_ncells(ncells)
        if self.save_flag and self.network_struct:
            cbmio.dump(
                self.network_struct.model_dump(),
                self.output_location(RecipeKeys.NETWORK_STRUCT),
                indent=4,
            )
        return self.network_struct

    def build_network(self, save_flag: bool=True):
        # Construct model
        net_builder = self.network_builder(self.network_struct)
        net_builder.build()
        if save_flag:
            net_builder.save(self.recipe_setup.network_dir)
        return net_builder

def netstruct_from_file(struct_file: pathlib.Path) -> structure.Network | None:
    network_struct = cbmio.load(struct_file)
    if network_struct:
        return structure.dict2netstruct(network_struct)
    return None
