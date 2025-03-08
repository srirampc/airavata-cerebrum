import itertools
import typing as t
from typing_extensions import override
import traitlets
#
from ..base import DbQuery, OpXFormer, XformItr

#
# Basic Transformers
#
class IdentityXformer(OpXFormer):
    @t.final
    class IdTraits(traitlets.HasTraits):
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
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.IdTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.IdTraits(**trait_values)


class TQDMWrapper(OpXFormer):
    @t.final
    class TqTraits(traitlets.HasTraits):
        jupyter = traitlets.Bool()

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
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.TqTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.TqTraits(**trait_values)


class DataSlicer(OpXFormer):
    @t.final
    class SliceTraits(traitlets.HasTraits):
        stop = traitlets.Int()
        list = traitlets.Bool()

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
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.SliceTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.SliceTraits(**trait_values)
#
#
def query_register() -> list[type[DbQuery]]:
    return []


def xform_register() -> list[type[OpXFormer]]:
    return [
        IdentityXformer,
        TQDMWrapper,
        DataSlicer
    ]
