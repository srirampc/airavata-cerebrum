import ast
import itertools
import logging
import typing as t
#
import duckdb
import polars as pl
import aisynphys
#
from typing_extensions import override
from pydantic import Field
from aisynphys.database import SynphysDatabase
from aisynphys.database.schema.experiment import PairBase
from aisynphys.cell_class import CellClass, classify_cells, classify_pairs
from aisynphys.connectivity import measure_connectivity
#
from ..base import (
    CerebrumBaseModel,
    BaseParams, 
    DbQuery,
    QryDBWriter,
    QryItr,
)


class CellClassSelection(t.NamedTuple):
    layer: str
    neuron: str
    criteria: dict[str, str | tuple[str, str]]

    @property
    def name(self) -> str:
        return self.layer + "-" + self.neuron


CELL_CLASS_SELECT = [
    CellClassSelection(
        "L23", "Pyr", {"dendrite_type": "spiny", "cortical_layer": "2/3"}
    ),
    CellClassSelection("L23", "Pvalb", {"cre_type": "pvalb", "cortical_layer": "2/3"}),
    CellClassSelection("L23", "Sst", {"cre_type": "sst", "cortical_layer": "2/3"}),
    CellClassSelection("L23", "Vip", {"cre_type": "vip", "cortical_layer": "2/3"}),
    CellClassSelection(
        "L4", "Pyr", {"cre_type": ("nr5a1", "rorb"), "cortical_layer": "4"}
    ),
    CellClassSelection("L4", "Pvalb", {"cre_type": "pvalb", "cortical_layer": "4"}),
    CellClassSelection("L4", "Sst", {"cre_type": "sst", "cortical_layer": "4"}),
    CellClassSelection("L4", "Vip", {"cre_type": "vip", "cortical_layer": "4"}),
    CellClassSelection(
        "L5", "ET", {"cre_type": ("sim1", "fam84b"), "cortical_layer": "5"}
    ),
    CellClassSelection("L5", "IT", {"cre_type": "tlx3", "cortical_layer": "5"}),
    CellClassSelection("L5", "Pvalb", {"cre_type": "pvalb", "cortical_layer": "5"}),
    CellClassSelection("L5", "Sst", {"cre_type": "sst", "cortical_layer": "5"}),
    CellClassSelection("L5", "Vip", {"cre_type": "vip", "cortical_layer": "5"}),
    CellClassSelection(
        "L6", "Pyr", {"cre_type": "ntsr1", "cortical_layer": ("6a", "6b")}
    ),
    CellClassSelection(
        "L6", "Pvalb", {"cre_type": "pvalb", "cortical_layer": ("6a", "6b")}
    ),
    CellClassSelection(
        "L6", "Sst", {"cre_type": "sst", "cortical_layer": ("6a", "6b")}
    ),
    CellClassSelection(
        "L6", "Vip", {"cre_type": "vip", "cortical_layer": ("6a", "6b")}
    ),
]

CELL_LAYER_SET = set([x.layer for x in CELL_CLASS_SELECT])
CELL_NEURON_SET = set([x.neuron for x in CELL_CLASS_SELECT])


def _log():
    return logging.getLogger(__name__)


class AISynPhysHelper:
    @staticmethod
    def select_cell_classes(
        layer_list: list[str] | None,
        neuron_list: list[str] | None = None
    ) -> dict[str, CellClass]:
        layer_set = CELL_LAYER_SET
        if layer_list:
            layer_set = set(layer_list)
        neuron_set = CELL_NEURON_SET
        if neuron_list:
            neuron_set = set(neuron_list)
        return {
            cselect.name: CellClass(name=cselect.name, **cselect.criteria)
            for cselect in CELL_CLASS_SELECT
            if (cselect.layer in layer_set) and (cselect.neuron in neuron_set)
        }
    
    @staticmethod
    def get_connectivity(
        layer_list: list[str] | None,
        qpairs : list[PairBase]
    ) -> tuple[dict[t.Any, t.Any], t.Literal[0, 1]]:
        cell_classes = AISynPhysHelper.select_cell_classes(layer_list)
        cell_groups = classify_cells(cell_classes.values(), pairs=qpairs)
        pair_groups = classify_pairs(qpairs, cell_groups)
        try:
            return measure_connectivity(
                pair_groups,
                sigma=100e-6,
                dist_measure="lateral_distance",
            ), 0 
        except RuntimeWarning as _rex:
            return {}, 1


class AISynInitParams(CerebrumBaseModel):
    download_base : t.Annotated[str, Field(title="Download Base Dir.")]
    projects : t.Annotated[list[str], Field(title="AI Syn. Projects")] = []
    db_size  : t.Annotated[str, Field(title="DB Size")] = "small"

class AISynExecParams(CerebrumBaseModel):
    layer : t.Annotated[list[str], Field(title="Layers")]

AISynBaseParams : t.TypeAlias = BaseParams[AISynInitParams, AISynExecParams]

class AISynPhysQuery(DbQuery[AISynInitParams, AISynExecParams]):
    class QryParams(AISynBaseParams):
        init_params: t.Annotated[AISynInitParams, Field(title='Init Params')]
        exec_params: t.Annotated[AISynExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: AISynInitParams, **_params: t.Any):
        """
        Initialize AI SynphysDatabase
        Parameters
        ----------
        download_base : str (Mandatory)
           File location to store the database
        projects : list[str] (optional)
           Run the database

        """
        self.name : str = __name__ + ".AIProject"
        self.download_base : str = init_params.download_base
        aisynphys.config.cache_path = self.download_base
        self.sdb : SynphysDatabase = SynphysDatabase.load_current(
            init_params.db_size
        )
        self.projects : list[str] =  (
            init_params.projects
            if init_params.projects else self.sdb.mouse_projects
        )
        self.qpairs : list[PairBase] = self.sdb.pair_query(
            project_name=self.projects
        ).all()
        self.nwarnings : int = 0

    @override
    def run(
        self,
        exec_params: AISynExecParams,
        first_iter: QryItr | None,
        *rest_iter: QryItr | None,
        **_params: t.Any,
    ) -> QryItr | None:
        """
        Get the connectivity probabilities for given layter

        Parameters
        ----------
        run_params: dict with the following keys:
            layer : List[str]
             list of layers of interest
        Returns
        -------
        dict of elements for each sub-regio:
        {
            subr 1 : {}
        }
        """
        #
        # default_args = {}
        # rarg = {**default_args, **params} if params else default_args
        _log().info("AISynPhysQuery Args : %s", str(exec_params))
        results, rerror = AISynPhysHelper.get_connectivity(
            exec_params.layer,
            self.qpairs
        )
        # cell_classes = self.select_cell_classes(layer_list)
        # cell_groups = classify_cells(cell_classes.values(), pairs=self.qpairs)
        # pair_groups = classify_pairs(self.qpairs, cell_groups)
        # nwarnings = 0
        # try:
        #     results = measure_connectivity(
        #         pair_groups, sigma=100e-6, dist_measure="lateral_distance"
        #     ) 
        # except RuntimeWarning:
        #     nwarnings += 1
        #     results = {}
        #     pass
        self.nwarnings += rerror
        _log().info(
            "AISynPhysQuery Args : [%s] ; N warnings [%d]",
            str(exec_params),
            self.nwarnings
        )
        #
        return [
            {
                repr(
                    (
                        x[0].name,
                        x[1].name,
                    )
                ): y["adjusted_connectivity"][0]
                for x, y in results.items()
            }
        ]

    @override
    @classmethod
    def params_type(cls) -> type[AISynBaseParams]:
        return cls.QryParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> AISynBaseParams:
        return cls.QryParams.model_validate(param_dict)


class DFBuilder:
    @staticmethod
    def syn_row(pair_literal: str, value: float) -> tuple[str, str, float]:
        pairx = ast.literal_eval(pair_literal)
        return (pairx[0], pairx[1], value)
    
    @staticmethod
    def split_pairs(p_dict: dict[str, t.Any]):
        return (
            DFBuilder.syn_row(px, vx)
            for px, vx in p_dict.items()
        )
 
    @staticmethod
    def build(
        in_iter: QryItr | None,
        **_params: t.Any,
    ) -> pl.DataFrame | None:
        if in_iter is None:
            return None
        rschema=[
            ("pre_synapse", pl.String),
            ("post_synapse", pl.String),
            ("connect_prob", pl.Float64)
        ]
        return pl.DataFrame(
            (
                itertools.chain.from_iterable(
                    DFBuilder.split_pairs(x) for x in in_iter
                )
            ),
            schema=rschema,
            orient="row",
        )


class AISynDuckDBWriter(QryDBWriter):
    def __init__(self, db_conn: duckdb.DuckDBPyConnection):
        self.conn : duckdb.DuckDBPyConnection = db_conn

    @override
    def write(
        self,
        in_iter: QryItr | None,
        **_params: t.Any,
    ) -> None:
        result_df = DFBuilder.build(in_iter) # pyright: ignore[reportUnusedVariable]
        self.conn.execute(
            "CREATE OR REPLACE TABLE ai_synphys AS SELECT * FROM result_df"
        )
        self.conn.commit()
