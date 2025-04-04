#
# Module for accesing databases
#

from ..base import DbQueryCBT, OpXFormerCBT, QryDBWriter

from .abc_mouse import (
    ABCDbMERFISH_CCFQuery,
    ABCDuckDBWriter 
)
from .abm_celltypes import (
    CTDbCellCacheQuery,
    CTDbCellApiQuery,
    CTDbGlifApiQuery,
    CTDbGlifApiModelConfigQry,
    ABMCTDuckDBWriter
)
from .ai_synphys import (
    AISynPhysQuery,
    AISynDuckDBWriter,
)
from .me_features import (
    MEFDataQuery,
    MEFDuckDBWriter,
)

#
# ------- Query and Xform Registers -----
#
def query_register() -> list[type[DbQueryCBT]]:
    return [
        ABCDbMERFISH_CCFQuery,
        CTDbCellCacheQuery,
        CTDbCellApiQuery,
        CTDbGlifApiQuery,
        CTDbGlifApiModelConfigQry,
        AISynPhysQuery,
        MEFDataQuery,
    ]


def xform_register() -> list[type[OpXFormerCBT]]:
    return []


def dbwriter_register() -> list[type[QryDBWriter]]:
    return [
        ABCDuckDBWriter,
        ABMCTDuckDBWriter,
        AISynDuckDBWriter,
        MEFDuckDBWriter,
    ]
