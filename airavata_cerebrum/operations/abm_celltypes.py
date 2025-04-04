import typing as t
import operator
#
from typing_extensions import override
from pydantic import Field
#
from ..base import (
    CerebrumBaseModel, NoneParams, BaseParams, 
    OpXFormer, XformItr
)
from .json_filter import (
    IJPExecParams, IJPtrExecParams,
    IterJPatchFilter, IterJPointerFilter
)
from .dict_filter import FilterPredicateT, IAFExecParams, IterAttrFilter
from ..util import flip_args


class CTMNExecParams(CerebrumBaseModel):
    name :t.Annotated[str, Field(title='Model Name')]

CTMNBaseParams : t.TypeAlias = BaseParams[NoneParams, CTMNExecParams]

class CTModelNameFilter(OpXFormer[NoneParams, CTMNExecParams]):
    class FilterParams(CTMNBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[CTMNExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **_params: t.Any):
        self.name : str = __name__ + ".CTModelNameFilter"
        self.filter_fmt : str = "$.glif.neuronal_models[?('{}' in @.name)]"
        self.dest_path : str = "/glif/neuronal_models"
        self.jpatch_filter : IterJPatchFilter = IterJPatchFilter(
            init_params, **_params
        )

    @override
    def xform(
        self,
        exec_params: CTMNExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any,
    ) -> XformItr | None:
        filter_exp = self.filter_fmt.format(exec_params.name)
        return self.jpatch_filter.xform(
            IJPExecParams(
                filter_exp=filter_exp,
                dest_path=self.dest_path,
            ),
            first_iter,
        )

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

class CTExplainedRatioFilter(OpXFormer[NoneParams, CTERExecParams]):
    class FilterParams(CTERBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[CTERExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.name : str = __name__ + ".CTExplainedRatioFilter"
        self.filter_fmt : str = "$.glif.neuronal_models[0].neuronal_model_runs[?(@.explained_variance_ratio > {})]"
        self.dest_path : str = "/glif/neuronal_models/0/neuronal_model_runs"
        self.final_path : str = "/glif/neuronal_models/0/neuronal_model_runs/0"
        self.jpatch_filter : IterJPatchFilter = IterJPatchFilter(
            init_params, **params
        )
        self.jptr_filter : IterJPointerFilter = IterJPointerFilter(
            init_params, **params
        )

    @override
    def xform(
        self,
        exec_params: CTERExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any,
    ) -> XformItr | None:
        filter_exp = self.filter_fmt.format(exec_params.ratio)
        patch_out = self.jpatch_filter.xform(
            IJPExecParams(filter_exp=filter_exp, dest_path=self.dest_path),
            first_iter,
        )
        return self.jptr_filter.xform(
            IJPtrExecParams(path=self.final_path),
            patch_out,
        )

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
    ] = None
    donor_in : t.Annotated[
        list[str] | None, Field(title="Donor in (line_name)")
    ] = None
    reporter_status : t.Annotated[
        str | None, Field(title="Reporter Status (cell_reporter_status)")
    ] = None 
    key    : t.Annotated[str , Field(title="Output Key")] = ""

CTPFBaseParams : t.TypeAlias = BaseParams[NoneParams, CTPFExecParams]

class CTPropertyFilter(OpXFormer[NoneParams, CTPFExecParams]):
    class FilterParams(CTPFBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[CTPFExecParams, Field(title='Exec Params')]


    QUERY_FILTER_MAP : dict[str, tuple[str, FilterPredicateT, type]] = {
        "region"     : ("structure_parent__acronym", operator.eq, str),
        "layer"      : ("structure__layer", operator.eq, str),
        "line"       : ("line_name", operator.contains, str),
        "line_start" : ("line_name", str.startswith, tuple),
        "donor_in"   : ("donor__id", flip_args(operator.contains), list),
        "reporter_status" : ("cell_reporter_status", operator.eq, str),
    }

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.cell_attr_filter : IterAttrFilter = IterAttrFilter(
            init_params, **params
        )

    def get_filter(
            self, 
            exec_params: CTPFExecParams,
            fkey: str
    ) -> tuple[str, FilterPredicateT, type] | None:
        pvalue = exec_params.get(fkey)
        if pvalue:
            attr, bin_op, rvtype = CTPropertyFilter.QUERY_FILTER_MAP[fkey]
            return (attr, bin_op, rvtype(pvalue))
        else:
            return None

    @override
    def xform(
        self,
        exec_params: CTPFExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any
    ) -> XformItr | None:
        filter_itr = (
            self.get_filter(exec_params, fkey)
            for fkey in CTPropertyFilter.QUERY_FILTER_MAP.keys()
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
    def params_type(cls) -> type[CTPFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> CTPFBaseParams:
        return cls.FilterParams.model_validate(param_dict)
