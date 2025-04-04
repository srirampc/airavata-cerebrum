
from ..base import DbQueryCBT, OpXFormerCBT
from .xform import (
    IdentityXformer,
    TQDMWrapper,
    DataSlicer,
)
from .abc_mouse import (
    ABCDbMERFISH_CCFFractionFilter,
    ABCDbMERFISH_CCFLayerRegionFilter,
)
from .abm_celltypes import (
    CTModelNameFilter,
    CTExplainedRatioFilter,
    CTPropertyFilter,
)
from .ai_synphys import AISynPhysPairFilter
from .dict_filter import (
    IterAttrFilter,
    IterAttrMapper,
)
from .json_filter import (
    IterJPointerFilter,
    IterJPatchFilter,
    JPointerFilter,
)
from .me_features import (
    MEPropertyFilter,
)


# ------- Query Registers -----
def query_register() -> list[type[DbQueryCBT]]:
    return []


# ------- Xform Registers -----
def xform_register() -> list[type[OpXFormerCBT]]:
    return [
        IdentityXformer,
        TQDMWrapper,
        DataSlicer,
        ABCDbMERFISH_CCFLayerRegionFilter,
        ABCDbMERFISH_CCFFractionFilter,
        CTModelNameFilter,
        CTExplainedRatioFilter,
        CTPropertyFilter,
        AISynPhysPairFilter,
        IterAttrMapper,
        IterAttrFilter,
        IterJPointerFilter,
        IterJPatchFilter,
        JPointerFilter,
        MEPropertyFilter,
    ]
