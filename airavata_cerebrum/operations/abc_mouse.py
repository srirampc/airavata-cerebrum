import typing as t
#
from typing_extensions import override
from pydantic import Field
#
from ..base import CerebrumBaseModel, OpXFormer, BaseParams, XformItr, DbQuery
from .json_filter import JPointerFilter


class ABFInitParams(CerebrumBaseModel):
    pass

class ABFExecParams(CerebrumBaseModel):
    region     : t.Annotated[str, Field(title="Region")]
    sub_region : t.Annotated[str, Field(title="Sub-region")]

ABFBaseParams : t.TypeAlias = BaseParams[ABFInitParams, ABFExecParams]

class ABCDbMERFISH_CCFLayerRegionFilter(OpXFormer):
    class FilterParams(ABFBaseParams):
        init_params: t.Annotated[ABFInitParams, Field(title='Init Params')]
        exec_params: t.Annotated[ABFExecParams, Field(title='Exec Params')]

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
    def params_type(cls) -> type[ABFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> ABFBaseParams:
        return cls.FilterParams.model_validate(param_dict)


class ABCCFInitParams(CerebrumBaseModel):
    pass

class ABCCFExecParams(CerebrumBaseModel):
    region   : t.Annotated[str, Field(title="Region")]
    cell_type : t.Annotated[str, Field(title="Cell Type")] = ""

ABCCFBaseParams : t.TypeAlias = BaseParams[ABCCFInitParams, ABCCFExecParams]

class ABCDbMERFISH_CCFFractionFilter(OpXFormer):
    class FilterParams(ABCCFBaseParams):
        init_params: t.Annotated[ABCCFInitParams, Field(title='Init Params')]
        exec_params: t.Annotated[ABCCFExecParams, Field(title='Exec Params')]

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
    def params_type(cls) -> type[ABCCFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> ABCCFBaseParams:
        return cls.FilterParams.model_validate(param_dict)


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
