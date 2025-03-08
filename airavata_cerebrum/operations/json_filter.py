import logging
import typing as t
from typing_extensions import override
import jsonpath
import traitlets
#
from ..base import OpXFormer, XformElt, XformItr, XformSeq


def _log():
    return logging.getLogger(__name__)


class JPointerFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        paths = traitlets.List()
        keys = traitlets.List()

    def __init__(self, **params: t.Any):
        self.name : str = __name__ + ".JPointerFilter"
        self.patch_out : str | None = None

    def resolve(self, fpath: str, dctx: XformSeq | None):
        jptr = jsonpath.JSONPointer(fpath)
        if dctx and jptr.exists(dctx):
            return jptr.resolve(dctx)
        else:
            return None

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        """
        Filter the output only if the destination path is present in dctx.

        Parameters
        ----------
        dctx : dict
           dictonary of ddescription
        params: requires the following keyword parameters
          {
            paths : List of JSON Path destination
            keys : Destination keys
          }

        Returns
        -------
        dctx: dict | None
           dict or None
        """
        fp_lst = params["paths"]
        key_lst = params["keys"]
        return [
            {key: self.resolve(
                fpath,
                in_iter  # pyright: ignore[reportArgumentType]
            ) for fpath, key in zip(fp_lst, key_lst)}
        ]

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.FilterTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.FilterTraits(**trait_values)


class IterJPatchFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        filter_exp = traitlets.Bytes()
        dest_path = traitlets.Bytes()

    def __init__(self, **init_params: t.Any):
        self.name : str = __name__ + ".IterJPatchFilter"
        self.patch_out : str | None = None

    def patch(self, ctx: XformElt, filter_exp: str, dest_path: str):
        fx = jsonpath.findall(filter_exp, ctx)
        try:
            if jsonpath.JSONPointer(dest_path).exists(ctx):
                return jsonpath.patch.apply(
                    [{"op": "replace", "path": dest_path, "value": fx}], ctx
                )
            else:
                return None
        except jsonpath.JSONPatchError as jpex:
            _log().error("Jpatch error : ", jpex)
            _log().debug(
                "Run arguments: Filter [%s]; Dest [%s]; Context [%s] ",
                filter_exp,
                dest_path,
                str(ctx),
            )
            return None

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        """
        Select the output matching the filter expression and place in
        the destination.

        Parameters
        ----------
        ct_iter : Iterator
           iterator of cell type descriptions
        filter_params: requires the following keyword parameters
          {
            filter_exp : JSON path filter expression
            dest_path : JSON Path destination
          }

        Returns
        -------
        ct_iter: iterator
           iterator of cell type descriptions
        """
        filter_exp = params["filter_exp"]
        dest_path = params["dest_path"]
        return (
            iter(self.patch(x, filter_exp, dest_path) for x in in_iter if x)
            if in_iter
            else None
        ) # pyright: ignore[reportReturnType]

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.FilterTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.FilterTraits(**trait_values)


class IterJPointerFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        path = traitlets.Unicode()

    def __init__(self, **params: t.Any):
        self.name : str = __name__ + ".IterJPointerFilter"
        self.patch_out : str | None = None

    def exists(self, ctx: XformElt, fpath: str):
        return jsonpath.JSONPointer(fpath).exists(ctx)

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        """
        Filter the output only if the destination path is present.

        Parameters
        ----------
        ct_iter : Iterator
           iterator of cell type descriptions
        params: requires the following keyword parameters
          {
            path : JSON Path destination
          }

        Returns
        -------
        ct_iter: iterator
           iterator of cell type descriptions
        """
        fpath = params["path"]
        return (
            iter(x for x in in_iter if x and self.exists(x, fpath)) if in_iter else None
        )

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.FilterTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.FilterTraits(**trait_values)


#
# ----- Mapper, Filter and Query Registers ------
#
def query_register() -> list[type]:
    return []


def xform_register():
    return [
        IterJPointerFilter,
        IterJPatchFilter,
        JPointerFilter,
    ]
