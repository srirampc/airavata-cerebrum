import typing as t
#
from typing_extensions import override
from pydantic import Field
#
from ..base import OpXFormer, BaseParams, XformItr, DbQuery
from .json_filter import JPointerFilter


class AISynPhysPairFilter(OpXFormer):
    class FilterParams(BaseParams):
        pre  : t.Annotated[str, Field(title='Pre-synapse')]
        post : t.Annotated[str, Field(title='Post-synapse')]

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
    def params_type(cls) -> type[BaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> BaseParams:
        return cls.FilterParams.model_validate(param_dict)

#
# ------- Query and Xform Registers -----
#
def query_register() -> list[type[DbQuery]]:
    return []


def xform_register() -> list[type[OpXFormer]]:
    return [
        AISynPhysPairFilter,
    ]
