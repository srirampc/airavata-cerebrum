import logging
import typing as t
from pydantic import Field
from typing_extensions import override
import jsonpath
#
from ..base import (
    CerebrumBaseModel,
    NoneParams, BaseParams,
    OpXFormer, XformItr,
    XformElt, XformSeq
)


def _log():
    return logging.getLogger(__name__)


class JPFExecParams(CerebrumBaseModel):
    paths : t.Annotated[list[str], Field(title="JSON Pointer Paths")]
    keys  : t.Annotated[list[str], Field(title="Keys")]

JPFBaseParams    : t.TypeAlias = BaseParams[NoneParams, JPFExecParams]

class JPointerFilter(OpXFormer[NoneParams, JPFExecParams]):
    class FilterParams(JPFBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[JPFExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
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
        exec_params: JPFExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any,
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
        fp_lst = exec_params.paths
        key_lst = exec_params.keys
        return [
            {key: self.resolve(
                fpath,
                first_iter  # pyright: ignore[reportArgumentType]
            ) for fpath, key in zip(fp_lst, key_lst)}
        ]

    @override
    @classmethod
    def params_type(cls) -> type[JPFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> JPFBaseParams:
        return cls.FilterParams.model_validate(param_dict)



class IJPExecParams(CerebrumBaseModel):
    filter_exp : t.Annotated[str, Field(title="Filter Expression")] # = traitlets.Bytes()
    dest_path  : t.Annotated[str, Field(title="Dest. Path")] # = traitlets.Bytes()

IJPBaseParams : t.TypeAlias = BaseParams[NoneParams, IJPExecParams]

class IterJPatchFilter(OpXFormer[NoneParams, IJPExecParams]):
    class FilterParams(IJPBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[IJPExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **_params: t.Any):
        self.name : str = __name__ + ".IterJPatchFilter"
        self.patch_out : str | None = None

    def patch(self, ctx: XformElt, filter_exp: str, dest_path: str) -> XformElt:
        fx = jsonpath.findall(filter_exp, ctx)
        try:
            if jsonpath.JSONPointer(dest_path).exists(ctx):
                return jsonpath.patch.apply(
                    [{"op": "replace", "path": dest_path, "value": fx}],
                    ctx
                ) # pyright: ignore[reportReturnType] TODO::Type match
            else:
                return {}
        except jsonpath.JSONPatchError as jpex:
            _log().error("Jpatch error : ", jpex)
            _log().debug(
                "Run arguments: Filter [%s]; Dest [%s]; Context [%s] ",
                filter_exp,
                dest_path,
                str(ctx),
            )
            return {}

    @override
    def xform(
        self,
        exec_params: IJPExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any,
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
        return (
            iter(
                self.patch(x, exec_params.filter_exp, exec_params.dest_path)
                for x in first_iter if x
            )
            if first_iter else {}
        )

    @override
    @classmethod
    def params_type(cls) -> type[IJPBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> IJPBaseParams:
        return cls.FilterParams.model_validate(param_dict)


class IJPtrExecParams(CerebrumBaseModel):
    path : t.Annotated[str, Field(title="JSON Path")]

IJPtrBaseParams : t.TypeAlias = BaseParams[NoneParams, IJPtrExecParams]

class IterJPointerFilter(OpXFormer[NoneParams, IJPtrExecParams]):
    class FilterParams(IJPtrBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[IJPtrExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **_params: t.Any):
        self.name : str = __name__ + ".IterJPointerFilter"
        self.patch_out : str | None = None

    def exists(self, ctx: XformElt, fpath: str):
        return jsonpath.JSONPointer(fpath).exists(ctx)

    @override
    def xform(
        self,
        exec_params: IJPtrExecParams,
        first_iter: XformItr | None,
        *rest_iter: XformItr | None,
        **_params: t.Any,
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
        return (
            iter(x for x in first_iter if x and self.exists(x, exec_params.path))
            if first_iter else None
        )

    @override
    @classmethod
    def params_type(cls) -> type[IJPtrBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> IJPtrBaseParams:
        return cls.FilterParams.model_validate(param_dict)
