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
class IdentityXformer(OpXFormer[NoneParams, NoneParams]):
    class IdParams(BaseParams[NoneParams, NoneParams]):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[NoneParams, Field(title='Exec Params')]

    @override
    def __init__(self, init_params: NoneParams, **params: t.Any):
        super().__init__(init_params, **params)

    @override
    def xform(
        self,
        exec_params: NoneParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        return first_iter

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
    jupyter : t.Annotated[bool, Field(title='Run in Jupyter Notebook')] = False

TQDMBaseParams : t.TypeAlias = BaseParams[NoneParams, TQDExecParams]

class TQDMWrapper(OpXFormer[NoneParams, TQDExecParams]):
    class TQDMParams(TQDMBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[TQDExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
        super().__init__(init_params, **params)

    @override
    def xform(
        self,
        exec_params: TQDExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any,
    ) -> XformItr | None:
        import tqdm.notebook
        if exec_params.jupyter:
            return tqdm.notebook.tqdm(first_iter)
        return tqdm.tqdm(first_iter)

    @override
    @classmethod
    def params_type(cls) -> type[TQDMBaseParams]:
        return cls.TQDMParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> TQDMBaseParams:
        return cls.TQDMParams.model_validate(param_dict)


class DSLExecParams(CerebrumBaseModel):
    stop : t.Annotated[int, Field(title='Stop')] = 10
    list : t.Annotated[bool, Field(title='Produce List Output')] = True

DSLBaseParams : t.TypeAlias = BaseParams[NoneParams, DSLExecParams]

class DataSlicer(OpXFormer[NoneParams, DSLExecParams]):
    class SliceParams(DSLBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[DSLExecParams, Field(title='Exec Params')]
    @override
    def __init__(self, init_params: NoneParams, **params: t.Any):
        super().__init__(init_params, **params)

    @override
    def xform(
        self,
        exec_params: DSLExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any,
    ) -> XformItr | None:
        if first_iter:
            ditr = itertools.islice(first_iter, exec_params.stop)
            return list(ditr) if exec_params.list else ditr

    @override
    @classmethod
    def params_type(cls) -> type[DSLBaseParams]:
        return cls.SliceParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> DSLBaseParams:
        return cls.SliceParams.model_validate(param_dict)
