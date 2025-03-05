import traitlets
import typing as t
from typing_extensions import override
#
from ..base import OpXFormer, XformItr, DbQuery
from .json_filter import JPointerFilter


class ABCDbMERFISH_CCFLayerRegionFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        region = traitlets.Unicode()
        sub_region = traitlets.Unicode()

    def __init__(self, **params: t.Any):
        self.jptr_filter : JPointerFilter = JPointerFilter(**params)
        self.path_fmt : str = "/0/{}/{}"

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params : t.Any
    ) -> XformItr | None:
        region = params["region"]
        sub_region = params["sub_region"]
        rpath = self.path_fmt.format(region, sub_region)
        return self.jptr_filter.xform(
            in_iter,
            paths=[rpath],
            keys=[sub_region],
        )

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.FilterTraits


class ABCDbMERFISH_CCFFractionFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        region = traitlets.Unicode()
        cell_type = traitlets.Unicode()

    def __init__(self, **params: t.Any):
        self.jptr_filter : JPointerFilter = JPointerFilter(**params)
        self.ifrac_fmt : str = "/0/{}/inhibitory fraction"
        self.fwr_fmt : str = "/0/{}/fraction wi. region"
        self.frac_fmt : str = "/0/{}/{} fraction"

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any
    ) -> XformItr | None:
        region = params["region"]
        frac_paths = [
            self.ifrac_fmt.format(region),
            self.fwr_fmt.format(region),
        ]
        frac_keys = ["inh_fraction", "region_fraction"]
        if "cell_type" in params and params["cell_type"]:
            cell_type = params["cell_type"]
            frac_paths.append(self.frac_fmt.format(region, cell_type))
            frac_keys.append("fraction")
        return self.jptr_filter.xform(
            in_iter,
            paths=frac_paths,
            keys=frac_keys,
        )

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.FilterTraits


#
# ------- Query and Xform Registers -----
#
def query_register() -> list[type[DbQuery]]:
    return []


def xform_register() -> list[type[OpXFormer]]:
    return [
        ABCDbMERFISH_CCFLayerRegionFilter,
        ABCDbMERFISH_CCFFractionFilter,
    ]
