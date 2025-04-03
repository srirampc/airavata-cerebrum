import typing as t
import operator
#
from typing_extensions import override
from pydantic import Field
#
from ..base import (CerebrumBaseModel, NoneParams, OpXFormer,
                    BaseParams, XformItr, DbQuery)
from .json_filter import IterJPatchFilter, IterJPointerFilter
from .dict_filter import IterAttrFilter
from ..util import flip_args


class CTMNExecParams(CerebrumBaseModel):
    name :t.Annotated[str, Field(title='Model Name')]

CTMNBaseParams : t.TypeAlias = BaseParams[NoneParams, CTMNExecParams]

class CTModelNameFilter(OpXFormer):
    class FilterParams(CTMNBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[CTMNExecParams, Field(title='Exec Params')]

    def __init__(self, **params: t.Any):
        self.name : str = __name__ + ".CTModelNameFilter"
        self.filter_fmt : str = "$.glif.neuronal_models[?('{}' in @.name)]"
        self.dest_path : str = "/glif/neuronal_models"
        self.jpatch_filter : IterJPatchFilter = IterJPatchFilter(**params)

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        name = params["name"]
        filter_exp = self.filter_fmt.format(name)
        return self.jpatch_filter.xform(in_iter,
                                        filter_exp=filter_exp,
                                        dest_path=self.dest_path)

    @override
    @classmethod
    def params_type(cls) -> type[CTMNBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> CTMNBaseParams:
        return cls.FilterParams.model_validate(param_dict)


class CTERExecParams(CerebrumBaseModel):
    ratio : t.Annotated[float, Field(title='Min. Ratio')]

CTERBaseParams : t.TypeAlias = BaseParams[NoneParams, CTERExecParams]

class CTExplainedRatioFilter(OpXFormer):
    class FilterParams(CTERBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[CTERExecParams, Field(title='Exec Params')]

    def __init__(self, **params: t.Any):
        self.name : str = __name__ + ".CTExplainedRatioFilter"
        self.filter_fmt : str = "$.glif.neuronal_models[0].neuronal_model_runs[?(@.explained_variance_ratio > {})]"
        self.dest_path : str = "/glif/neuronal_models/0/neuronal_model_runs"
        self.final_path : str = "/glif/neuronal_models/0/neuronal_model_runs/0"
        self.jpatch_filter : IterJPatchFilter = IterJPatchFilter(**params)
        self.jptr_filter : IterJPointerFilter = IterJPointerFilter(**params)

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        ratio_value = params["ratio"]
        filter_exp = self.filter_fmt.format(ratio_value)
        patch_out = self.jpatch_filter.xform(in_iter,
                                             filter_exp=filter_exp,
                                             dest_path=self.dest_path)
        return self.jptr_filter.xform(patch_out,
                                      path=self.final_path)

    @override
    @classmethod
    def params_type(cls) -> type[CTERBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> CTERBaseParams:
        return cls.FilterParams.model_validate(param_dict)


class CTPFExecParams(CerebrumBaseModel):
    region : t.Annotated[
        str | None, Field(title="Region (structure_parent__acronym)")
    ] = None
    layer  : t.Annotated[
        str | None, Field(title="Layer (structure__layer)")
    ] = None
    line   : t.Annotated[
        str | None, Field(title="Line (line_name)")
    ] = None
    line_start : t.Annotated[
        tuple[str] | None, Field(title="Lines Starts With (line_name)")
    ]
    reporter_status : t.Annotated[
        str | None, Field(title="Reporter Status (cell_reporter_status)")
    ] = None 
    key    : t.Annotated[str , Field(title="Output Key")] = ""

CTPFBaseParams : t.TypeAlias = BaseParams[NoneParams, CTPFExecParams]

class CTPropertyFilter(OpXFormer):
    class FilterParams(CTPFBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[CTPFExecParams, Field(title='Exec Params')]

    PredicateT : t.TypeAlias = t.Callable[[t.Any, t.Any], bool]

    QUERY_FILTER_MAP : dict[str, tuple[str, PredicateT, type]] = {
        "region"     : ("structure_parent__acronym", operator.eq, str),
        "layer"      : ("structure__layer", operator.eq, str),
        "line"       : ("line_name", operator.contains, str),
        "line_start" : ("line_name", str.startswith, tuple),
        "donor_in"   : ("donor__id", flip_args(operator.contains), list),
        "reporter_status" : ("cell_reporter_status", operator.eq, str),
    }

    def __init__(self, **params: t.Any):
        self.cell_attr_filter : IterAttrFilter = IterAttrFilter(**params)

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any
    ) -> XformItr | None:
        key = params["key"] if "key" in params and params["key"] else None
        filters = []
        for pkey, valx in params.items():
            if pkey in CTPropertyFilter.QUERY_FILTER_MAP:
                attr, bin_op, rvtype = CTPropertyFilter.QUERY_FILTER_MAP[pkey]
                filters.append([attr, bin_op, rvtype(valx)])
        return self.cell_attr_filter.xform(
            in_iter,
            key=key,
            filters=filters
        )

    @override
    @classmethod
    def params_type(cls) -> type[CTPFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> CTPFBaseParams:
        return cls.FilterParams.model_validate(param_dict)

#
# ------- Query and Xform Registers -----
#
def query_register() -> list[type[DbQuery]]:
    return []


def xform_register() -> list[type[OpXFormer]]:
    return [
        CTModelNameFilter,
        CTExplainedRatioFilter,
        CTPropertyFilter,
    ]
