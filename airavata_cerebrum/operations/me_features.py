import typing as t
import operator

from typing_extensions import override
from pydantic import Field

#
from ..base import (
    CerebrumBaseModel, BaseParams,  NoneParams,
    OpXFormer, XformItr
)
from ..util import flip_args
from .dict_filter import FilterPredicateT, IAFExecParams, IAFInitParams, IterAttrFilter


class MEFExecParams(CerebrumBaseModel):
    key: t.Annotated[str, Field(title="Key (iter key)")] = "mef"
    me_contains : t.Annotated[
        list[str] | None, Field(title="ME Type (me-type)")
    ] = None
    e_contains : t.Annotated[
        list[str] | None, Field(title="E Type (e-type)")
    ] = None

MEFBaseParams : t.TypeAlias = BaseParams[NoneParams, MEFExecParams]

class MEPropertyFilter(OpXFormer[NoneParams, MEFExecParams]):
    class FilterParams(MEFBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[MEFExecParams, Field(title='Exec Params')]

    QUERY_FILTER_MAP : dict[str, tuple[str, FilterPredicateT, type]] = {
        "me_contains"   : ("me-type", flip_args(operator.contains), list),
        "e_contains" : ("e-type", flip_args(operator.contains), list),
    }

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.cell_attr_filter : IterAttrFilter = IterAttrFilter(
            IAFInitParams(combine="all"), **params
        )

    def get_filter(
            self, 
            exec_params: MEFExecParams,
            fkey: str
    ) -> tuple[str, FilterPredicateT, type] | None:
        pvalue = exec_params.get(fkey)
        if pvalue:
            attr, bin_op, rvtype = MEPropertyFilter.QUERY_FILTER_MAP[fkey]
            return (attr, bin_op, rvtype(pvalue))
        else:
            return None

    @override
    def xform(
        self,
        exec_params: MEFExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any
    ) -> XformItr | None:
        filter_itr = (
            self.get_filter(exec_params, fkey)
            for fkey in MEPropertyFilter.QUERY_FILTER_MAP.keys()
        )
        filters = [fx for fx in filter_itr if fx]
        return self.cell_attr_filter.xform(
            IAFExecParams(
                key=exec_params.key,
                filters=filters,
            ),
            first_iter,
        )

    @override
    @classmethod
    def params_type(cls) -> type[MEFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> MEFBaseParams:
        return cls.FilterParams.model_validate(param_dict)
