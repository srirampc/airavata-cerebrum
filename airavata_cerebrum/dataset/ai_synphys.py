import logging
import typing as t
import traitlets
import aisynphys
#
from typing_extensions import override
from aisynphys.database import SynphysDatabase
from aisynphys.database.schema.experiment import PairBase
from aisynphys.cell_class import CellClass, classify_cells, classify_pairs
from aisynphys.connectivity import measure_connectivity
#
from ..base import DbQuery, OpXFormer, QryItr


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


class AISynPhysQuery(DbQuery):
    @t.final
    class QryTraits(traitlets.HasTraits):
        download_base = traitlets.Unicode()
        layer = traitlets.List()

    def __init__(self, **params: t.Any):
        """
        Initialize AI SynphysDatabase
        Parameters
        ----------
        download_base : str (Mandatory)
           File location to store the database
        projects : list[str] (optional)
           Run the database

        """
        self.name : str = __name__ + ".ABCDbMERFISHQuery"
        self.download_base : str = params["download_base"]
        aisynphys.config.cache_path = self.download_base
        self.sdb : SynphysDatabase = SynphysDatabase.load_current("small")
        self.projects : list[str] = self.sdb.mouse_projects
        if "projects" in params and params["projects"]:
            self.projects = params["projects"]
        self.qpairs : list[PairBase] = self.sdb.pair_query(project_name=self.projects).all()

    def select_cell_classes(
        self,
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

    @override
    def run(
        self,
        in_iter: QryItr | None,
        **params: t.Any,
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
        default_args = {}
        rarg = {**default_args, **params} if params else default_args
        _log().info("AISynPhysQuery Args : %s", rarg)
        layer_list = rarg["layer"]
        cell_classes = self.select_cell_classes(layer_list)
        cell_groups = classify_cells(cell_classes.values(), pairs=self.qpairs)
        pair_groups = classify_pairs(self.qpairs, cell_groups)
        nwarnings = 0
        try:
            results = measure_connectivity(
                pair_groups, sigma=100e-6, dist_measure="lateral_distance"
            ) 
        except RuntimeWarning:
            nwarnings += 1
            results = {}
            pass
        _log().info("AISynPhysQuery Args : [%s] ; N warnings [%d]", rarg, nwarnings)
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
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.QryTraits


#
# ------- Query and Xform Registers -----
#
def query_register() -> list[type[DbQuery]]:
    return [
        AISynPhysQuery,
    ]


def xform_register() -> list[type[OpXFormer]]:
    return []
