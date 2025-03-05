import typing as t
import traitlets
from typing_extensions import override
#
from ..base import OpXFormer, XformItr, DbQuery
from .json_filter import JPointerFilter


class AISynPhysPairFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        pre = traitlets.Unicode()
        post = traitlets.Unicode()

    def __init__(self, **params: t.Any):
        self.jptr_filter: JPointerFilter = JPointerFilter(**params)
        self.path_fmt : str = "/0/{}"

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        npre = params["pre"] if "pre" in params else None
        npost = params["post"] if "post" in params else None
        rpath = self.path_fmt.format(repr((npre, npost)))
        return self.jptr_filter.xform(in_iter, paths=[rpath], keys=["probability"])

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
        AISynPhysPairFilter,
    ]
