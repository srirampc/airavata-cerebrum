import argparse
import logging
import operator
import typing as t
import codetiming
import pydantic
import pandas as pd
#
from pathlib import Path
#
#
import mousev1.model as v1model
import mousev1.operations as v1ops
import mousev1.mpi_utils as mpi_utils
from airavata_cerebrum.recipe import ModelRecipe, RecipeSetup
from airavata_cerebrum.model.structure import Network
from airavata_cerebrum.util.io import load as loadio
#
from codetiming import Timer

def _log():
    return logging.getLogger(__name__)


class RcpSettings(pydantic.BaseModel):
    name: str = "v1"
    ncells: int = 120000
    base_dir: Path = Path("./")
    recipe_dir: Path = Path("./model_recipes/v1")
    recipe_output_dir: Path | None = Path("./model_data/v1")
    levels: list[str] = ["L1",  "L23", "L4", "L5", "L6"]
    recipe_levels: dict[str, str | Path] = {
        "L1"  : "recipe_dm_l1.json",
        "L23" : "recipe_dm_l23.json",
        "L4"  : "recipe_dm_l4.json",
        "L5"  : "recipe_dm_l5.json",
        "L6"  : "recipe_dm_l6.json",
    }
    custom_mod_main : Path = Path("./model_recipes/v1/custom_mod.json")
    custom_mod_levels: dict[str, str | Path] = {
        "L1"  : Path("./model_recipes/v1/custom_mod_l1.json"),
        "L23" : Path("./model_recipes/v1/custom_mod_l23.json"),
        "L4"  : Path("./model_recipes/v1/custom_mod_l4.json"),
        "L5"  : Path("./model_recipes/v1/custom_mod_l5.json"),
        "L6"  : Path("./model_recipes/v1/custom_mod_l6.json"),
    }
    custom_mod_exts : list[str | Path] = [
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
        recipe_output_dir=rcp_set.output_dir,
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
    times_df = pd.DataFrame(data=[{
        "name": name,
        "ncalls": Timer.timers.count(name),
        "total_time": ttime,
        "min_time": Timer.timers.min(name),
        "max_time": Timer.timers.max(name),
        "mean_time": Timer.timers.mean(name),
        "median_time": Timer.timers.median(name),
        "stdev_time": Timer.timers.stdev(name),
    } for name, ttime in sorted(
        Timer.timers.items(),
        key=operator.itemgetter(1),
        reverse=True
    )])
    # set display options to show all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # print the dataframe
    print(times_df)

def summarize_timeings() -> list[dict[str, t.Any]]:
    rtimers = codetiming.Timer.timers
    times_rs = [{
        "name": name,
        "proc": mpi_utils.mpi_rank,
        "ncalls": rtimers.count(name),
        "total_time": ttime,
        "min_time": rtimers.min(name),
        "max_time": rtimers.max(name),
        "mean_time": rtimers.mean(name),
        "median_time": rtimers.median(name),
        "stdev_time": rtimers.stdev(name),
    } for name, ttime in sorted(
        rtimers.items(),
        key=operator.itemgetter(1),
        reverse=True
    )]
    collect_times_rs = mpi_utils.collect_merge_lists_at_root(times_rs)
    if mpi_utils.mpi_rank > 0 or not collect_times_rs:
        return [{}]
    # rtimers = alltimers[0]
    return collect_times_rs


def struct_mpi_bmtk(rcp_set: RcpSettings):
    rcp_set.save_flag = False
    md_dex = model_recipe(rcp_set)
    md_dex.build_net_struct()
    md_dex.apply_mod(rcp_set.ncells)
    md_dex.build_network()
    times_df = pd.DataFrame(data=summarize_timeings())
    if mpi_utils.mpi_rank == 0:
        # set display options to show all columns and rows
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        # print the dataframe
        print(times_df)



def build_bmtk(rcp_set: RcpSettings):
    md_dex = model_recipe(rcp_set)
    md_dex.download_db_data()
    md_dex.run_db_post_ops()
    md_dex.map_source_data()
    md_dex.build_net_struct()
    md_dex.apply_mod(rcp_set.ncells)
    md_dex.build_network()

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
    from v1_bmtk_simulate import load_nest_sonata
    return load_nest_sonata()

def main(input_config_file: str):
    if mpi_utils.mpi_rank == 0:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)
    cfg_set = RcpSettings.model_validate(loadio(input_config_file))
    _log().info(cfg_set.model_dump_json(indent=4))
    # convert_models_to_nest(cfg_set)
    # build_bmtk(cfg_set)
    if mpi_utils.mpi_size == 1:
        struct_bmtk(cfg_set)
    else:
        struct_mpi_bmtk(cfg_set)


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
