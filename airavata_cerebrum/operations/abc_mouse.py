import typing as t
#
from typing_extensions import override
from pydantic import Field
#
from ..base import (CerebrumBaseModel, NoneParams, OpXFormer,
                    BaseParams, XformItr)
from .json_filter import JPFExecParams, JPointerFilter


class ABFExecParams(CerebrumBaseModel):
    region     : t.Annotated[str, Field(title="Region")]
    sub_region : t.Annotated[str, Field(title="Sub-region")]

ABFBaseParams : t.TypeAlias = BaseParams[NoneParams, ABFExecParams]

class ABCDbMERFISH_CCFLayerRegionFilter(OpXFormer[NoneParams, ABFExecParams]):
    class FilterParams(ABFBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[ABFExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.jptr_filter : JPointerFilter = JPointerFilter(
            init_params, **params
        )
        self.path_fmt : str = "/0/{}/{}"

    @override
    def xform(
        self,
        exec_params: ABFExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params : t.Any
    ) -> XformItr | None:
        region = exec_params.region
        sub_region = exec_params.sub_region
        rpath = self.path_fmt.format(region, sub_region)
        return self.jptr_filter.xform(
            JPFExecParams(
                paths=[rpath],
                keys=[sub_region],
            ),
            first_iter,
        )

    @override
    @classmethod
    def params_type(cls) -> type[ABFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> ABFBaseParams:
        return cls.FilterParams.model_validate(param_dict)


class ABCCFExecParams(CerebrumBaseModel):
    region   : t.Annotated[str, Field(title="Region")]
    cell_type : t.Annotated[str, Field(title="Cell Type")] = ""

ABCCFBaseParams : t.TypeAlias = BaseParams[NoneParams, ABCCFExecParams]

class ABCDbMERFISH_CCFFractionFilter(OpXFormer[NoneParams, ABCCFExecParams]):
    class FilterParams(ABCCFBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[ABCCFExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.jptr_filter : JPointerFilter = JPointerFilter(
            init_params, **params
        )
        self.ifrac_fmt : str = "/0/{}/inhibitory fraction"
        self.fwr_fmt : str = "/0/{}/fraction wi. region"
        self.frac_fmt : str = "/0/{}/{} fraction"

    @override
    def xform(
        self,
        exec_params: ABCCFExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any
    ) -> XformItr | None:
        region = exec_params.region
        frac_paths = [
            self.ifrac_fmt.format(region),
            self.fwr_fmt.format(region),
        ]
        frac_keys = ["inh_fraction", "region_fraction"]
        if exec_params.cell_type:
            cell_type = exec_params.cell_type
            frac_paths.append(self.frac_fmt.format(region, cell_type))
            frac_keys.append("fraction")
        return self.jptr_filter.xform(
            JPFExecParams(
                paths=frac_paths,
                keys=frac_keys,
            ),
            first_iter,
        )

    @override
    @classmethod
    def params_type(cls) -> type[ABCCFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> ABCCFBaseParams:
        return cls.FilterParams.model_validate(param_dict)
