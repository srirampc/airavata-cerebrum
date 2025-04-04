import typing as t
#
from typing_extensions import override
from pydantic import Field
#
from ..base import (CerebrumBaseModel, NoneParams, OpXFormer,
                    BaseParams, XformItr)
from .json_filter import JPFExecParams, JPointerFilter


class AISPPExecParams(CerebrumBaseModel):
    pre  : t.Annotated[str, Field(title='Pre-synapse')]
    post : t.Annotated[str, Field(title='Post-synapse')]

AISPPBaseParams : t.TypeAlias = BaseParams[NoneParams, AISPPExecParams]

class AISynPhysPairFilter(OpXFormer[NoneParams, AISPPExecParams]):
    class FilterParams(AISPPBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[AISPPExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.jptr_filter: JPointerFilter = JPointerFilter(
            init_params, **params
        )
        self.path_fmt : str = "/0/{}"

    @override
    def xform(
        self,
        exec_params: AISPPExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any,
    ) -> XformItr | None:
        npre = exec_params.pre
        npost = exec_params.post
        rpath = self.path_fmt.format(repr((npre, npost)))
        return self.jptr_filter.xform(
            JPFExecParams(paths=[rpath], keys=["probability"]),
            first_iter,
        )

    @override
    @classmethod
    def params_type(cls) -> type[AISPPBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> AISPPBaseParams:
        return cls.FilterParams.model_validate(param_dict)
