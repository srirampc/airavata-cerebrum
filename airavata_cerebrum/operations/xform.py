import itertools
import typing as t
#
from pydantic import Field
from typing_extensions import override
#
from ..base import CerebrumBaseModel, DbQuery, NoneParams, OpXFormer, BaseParams, XformItr

#
# Basic Transformers
#
class IdentityXformer(OpXFormer):
    class IdParams(BaseParams[NoneParams, NoneParams]):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[NoneParams, Field(title='Exec Params')]

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        return in_iter

    @override
    @classmethod
    def params_type(cls) -> type[BaseParams[NoneParams, NoneParams]]:
        return cls.IdParams

    @override
    @classmethod
    def params_instance(
        cls,
        param_dict: dict[str, t.Any]
    ) -> BaseParams[NoneParams, NoneParams]:
        return cls.IdParams.model_validate(param_dict)

class TQDExecParams(CerebrumBaseModel):
    jupyter : t.Annotated[bool, Field(title='Run in Jupyter Notebook')]

TQDMBaseParams : t.TypeAlias = BaseParams[NoneParams, TQDExecParams]

class TQDMWrapper(OpXFormer):
    class TQDMParams(TQDMBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[TQDExecParams, Field(title='Exec Params')]

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        import tqdm.notebook
        if "jupyter" in params and params["jupyter"]:
            return tqdm.notebook.tqdm(in_iter)
        return tqdm.tqdm(in_iter)

    @override
    @classmethod
    def params_type(cls) -> type[TQDMBaseParams]:
        return cls.TQDMParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> TQDMBaseParams:
        return cls.TQDMParams.model_validate(param_dict)


class DSLExecParams(CerebrumBaseModel):
    stop : t.Annotated[int, Field(title='Stop')]
    list : t.Annotated[bool, Field(title='Produce List Output')]

DSLBaseParams : t.TypeAlias = BaseParams[NoneParams, DSLExecParams]

class DataSlicer(OpXFormer):
    class SliceParams(DSLBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[DSLExecParams, Field(title='Exec Params')]

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        default_args = {"stop": 10, "list": True}
        rarg = default_args | params if params else default_args
        if in_iter:
            ditr = itertools.islice(in_iter, rarg["stop"])
            return list(ditr) if bool(rarg["list"]) else ditr

    @override
    @classmethod
    def params_type(cls) -> type[DSLBaseParams]:
        return cls.SliceParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> DSLBaseParams:
        return cls.SliceParams.model_validate(param_dict)


def query_register() -> list[type[DbQuery]]:
    return []


def xform_register() -> list[type[OpXFormer]]:
    return [
        IdentityXformer,
        TQDMWrapper,
        DataSlicer
    ]
