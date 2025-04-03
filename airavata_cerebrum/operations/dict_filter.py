import logging
import typing as t
from pydantic import Field
from typing_extensions import override
#
from ..base import (CerebrumBaseModel, NoneParams, OpXFormer,
                    BaseParams, XformItr, XformElt, DbQuery)


def _log():
    return logging.getLogger(__name__)


class IAMInitParams(CerebrumBaseModel):
    attribute : t.Annotated[str, Field(title='Attribute Selected to Map')]

IAMBaseParams : t.TypeAlias = BaseParams[IAMInitParams, NoneParams]

class IterAttrMapper(OpXFormer):
    class MapperParams(IAMBaseParams):
        init_params: t.Annotated[IAMInitParams, Field(title='Init Params')]
        exec_params: t.Annotated[NoneParams, Field(title='Exec Params')]

    def __init__(self, **params: t.Any):
        """
        Attribute value mapper

        Parameters
        ----------
        attribute : str
           Attribute of the cell type; key of the cell type descr. dict
        """
        self.name : str = __name__ + ".IterAttrMapper"
        self.attr: str = params["attribute"]

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any
    ) -> XformItr | None:
        """
        Get values from cell type descriptions

        Parameters
        ----------
        in_iter : Iterator
           iterator of cell type descriptions
        attribute : str
           Attribute of the cell type; key of the cell type descr. dict

        Returns
        -------
        value_iter: iterator
           iterator of values from cell type descriptions for given attribute
        """
        if not in_iter:
            return None
        return iter(x[self.attr] for x in in_iter)

    @override
    @classmethod
    def params_type(cls) -> type[IAMBaseParams]:
        return cls.MapperParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> IAMBaseParams:
        return cls.MapperParams.model_validate(param_dict)

class IAFInitParams(CerebrumBaseModel):
    combine : t.Literal['any', 'all'] = 'all'

class IAFExecParams(CerebrumBaseModel):
    key     : t.Annotated[str, Field(title='Key')]
    filters : t.Annotated[list[tuple[t.Any, ...]], Field(title='Filters')]

IAFBaseParams : t.TypeAlias = BaseParams[NoneParams, IAFExecParams]

class IterAttrFilter(OpXFormer):
    class FilterParams(IAFBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[IAFExecParams, Field(title='Exec Params')]

    PREDICATE_COMBINE : dict[t.Literal['any', 'all'], t.Callable[[t.Any], bool]] = {
        'any' : any,
        'all' : all
    }

    def __init__(
            self,
            combine: t.Literal['any', 'all'] = 'all',
            **params: t.Any
    ):
        self.name : str = __name__ + ".IterAttrFilter"
        self.select_fn : t.Callable[[XformElt], XformElt] = lambda x: x
        self.combine_fn : t.Callable[[t.Any], bool] = (
            IterAttrFilter.PREDICATE_COMBINE[combine]
        )

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        key: str = '',
        filters: list[tuple[t.Any, ...]] | None = None,
        **params: t.Any,
    ) -> XformItr | None:
        """
        Filter cell type descriptions matching all the values for the given attrs.

        Parameters
        ----------
        ct_iter : Iterator
           iterator of cell type descriptions
        filter_params: requires the following keyword parameters
        {
            filters (mandatory): Iterater of triples [attribute, bin_op, value]
                attribute : attribute of the cell type;
                  key of the cell type descr. dict
                bin_op    : binary operation special function (mandatory)
                value     : attributes acceptable value (mandatory)
                  value in the cell type descr. dict
        }

        Returns
        -------
        ct_iter: iterator
           iterator of cell type descriptions
        """
        _log().info("IterAttrFilter Args : %s", params)
        filters_itr : list[tuple[t.Any, ...]] = filters if filters else []
        if key:
            self.select_fn = lambda x: x[params["key"]]
        return iter(
            rcdx
            for rcdx in in_iter
            if rcdx and self.combine_fn(
                bin_predicate(self.select_fn(rcdx)[attr], val)
                for attr, bin_predicate, val in filters_itr
            )
        ) if in_iter else None

    @override
    @classmethod
    def params_type(cls) -> type[IAFBaseParams]:
        return cls.FilterParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> IAFBaseParams:
        return cls.FilterParams.model_validate(param_dict)


#
# ------- Query and Xform Registers -----
#
def query_register() -> list[type[DbQuery]]:
    return []


def xform_register() -> list[type[OpXFormer]]:
    return [
        IterAttrMapper,
        IterAttrFilter,
    ]
