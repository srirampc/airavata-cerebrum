import itertools
import typing as t
#
from pydantic import Field
from typing_extensions import override
#
from ..base import DbQuery, OpXFormer, BaseParams, XformItr

#
# Basic Transformers
#
class IdentityXformer(OpXFormer):
    class IdParams(BaseParams):
        pass

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        return in_iter

    @override
    @classmethod
    def params_type(cls) -> type[BaseParams]:
        return cls.IdParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> BaseParams:
        return cls.IdParams.model_validate(param_dict)


class TQDMWrapper(OpXFormer):
    class TQDMParams(BaseParams):
        jupyter : t.Annotated[bool, Field(title='Run in Jupyter Notebook')]

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
    def params_type(cls) -> type[BaseParams]:
        return cls.TQDMParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> BaseParams:
        return cls.TQDMParams.model_validate(param_dict)


class DataSlicer(OpXFormer):
    class SliceParams(BaseParams):
        stop : t.Annotated[int, Field(title='Stop')]
        list : t.Annotated[bool, Field(title='Produce List Output')]

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
    def params_type(cls) -> type[BaseParams]:
        return cls.SliceParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> BaseParams:
        return cls.SliceParams.model_validate(param_dict)


def query_register() -> list[type[DbQuery]]:
    return []


def xform_register() -> list[type[OpXFormer]]:
    return [
        IdentityXformer,
        TQDMWrapper,
        DataSlicer
    ]
