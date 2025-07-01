import argparse
import logging
import typing as t
import pydantic
#
from pathlib import Path
from codetiming import Timer, TimerConfig
from bmtk.builder.network_adaptors.dm_network import DenseNetwork
#
from airavata_cerebrum.recipe import ModelRecipe, RecipeSetup
from airavata_cerebrum.model.structure import Network
from airavata_cerebrum.util.io import load as loadio
#
from mousev1.model import (
    V1BMTKNetworkBuilder,
    V1ConnectionMapper,
    V1NeuronMapper,
    V1RegionMapper
)
from mousev1.operations import (
    convert_ctdb_models_to_nest
)
from mousev1.comm_interface import (
    default_comm,
    CommInterface
)
from mousev1.dm_network import(
    MVParMethod,
    MVDenseNetwork
)


def _log():
    return logging.getLogger(__name__)

NetworkConstructor: t.TypeAlias = t.Literal['DenseNetwork', 'MVDenseNetwork'] 

def get_build_adaptor(nc: NetworkConstructor):
    match nc:
        case 'DenseNetwork':
            return DenseNetwork
        case "MVDenseNetwork":
            return MVDenseNetwork


class CerebrumRecipeRunner(pydantic.BaseModel):
    name: str = "v1"
    ncells: int = 120000
    enable_timing: bool = True
    logging_level: int = logging.DEBUG
    root_logging_level: int = logging.DEBUG
    build_adaptor_class: t.Literal['DenseNetwork', 'MVDenseNetwork'] = 'DenseNetwork'
    build_parallel_method: MVParMethod = MVParMethod.ALL_GATHER_BY_SND_RCV
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

    @Timer(name="RecipeSetupConfig.cerebrum_struct", logger=None)
    def cerebrum_struct(self) -> ModelRecipe:
        mdr = self.model_recipe()
        mdr.build_net_struct()
        mdr.apply_mod(self.ncells)
        return mdr

    def cerebrum_workflow(self):
        md_dex = self.model_recipe()
        md_dex.download_db_data()
        md_dex.run_db_post_ops()
        md_dex.map_source_data()
        md_dex.build_net_struct()
        md_dex.apply_mod(self.ncells)
        md_dex.build_network(
            adaptor_cls=get_build_adaptor(self.build_adaptor_class),
            parallel_method=self.build_parallel_method,
        )

    def convert_models_to_nest(self):
        convert_ctdb_models_to_nest(
            str(self.ctdb_models_dir), 
            str(self.nest_models_dir)
        )


@Timer(name="v1_build.mdr_build", logger=None)
def mdr_build(mdr:ModelRecipe, rcp_set: CerebrumRecipeRunner):
    mdr.build_network(
        adaptor_cls=get_build_adaptor(rcp_set.build_adaptor_class),
        parallel_method=rcp_set.build_parallel_method,
    )


def struct_bmtk(rcp_set: CerebrumRecipeRunner, comm: CommInterface):
    # Dont save it to db when running with MPI
    if comm.size > 0:
        rcp_set.save_flag = False
    mdrcp = rcp_set.cerebrum_struct()
    mdr_build(mdrcp, rcp_set)
    comm.log_profile_summary(
        _log(),
        logging.DEBUG, 
        f"{mdrcp.recipe_setup.base_dir}/runtimes_{comm.size}.csv",
        f"{mdrcp.recipe_setup.base_dir}/mem_used_{comm.size}.csv",
    )


def data_mapped_model(levels: tuple[str,...]=("L1", "L23", "L4")):
    rcp_set = CerebrumRecipeRunner(levels=list(levels))
    mdrcp = rcp_set.model_recipe()
    mdrcp.map_source_data()
    mdrcp.build_net_struct()
    mdrcp.apply_mod(rcp_set.ncells)
    mdrcp.build_network()


def run_sonata():
    from v1_bmtk_simulate import load_nest_sonata
    return load_nest_sonata()

def main(input_config_file: str):
    cfg_set = CerebrumRecipeRunner.model_validate(loadio(input_config_file))
    comm = default_comm()
    if comm.rank == 0:
        logging.basicConfig(level=cfg_set.root_logging_level)
    else:
        logging.basicConfig(level=cfg_set.logging_level)
    comm.log_at_root(_log(), logging.INFO, cfg_set.model_dump_json(indent=4))
    # cfg_set.convert_models_to_nest()
    # build_bmtk(cfg_set)
    if cfg_set.enable_timing:
        TimerConfig.enable_timers()
    else:
        TimerConfig.disable_timers()
    #
    struct_bmtk(cfg_set, comm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build V1 SONATA network file."
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="YAML netork setup.",
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
