import abc
import typing as t
#
from collections.abc import Iterable, Sequence
from typing_extensions import Self, override
from pydantic import BaseModel, Field


#
# pydantic-based BaseModel for Cerebrum
class CerebrumBaseModel(BaseModel):
    name : t.Annotated[str, Field(title="Name")] = ""

    @override
    def exclude(self) -> set[str]:
        return set(["name"])
 
    def is_valid_field(self, field: str) -> bool:
        return field in self.model_fields

    def get(self, field: str, val_not_found: t.Any = None) -> t.Any:
        try:
            return getattr(self, field)
        except AttributeError:
            return val_not_found


INPGT = t.TypeVar('INPGT', bound='CerebrumBaseModel')
EXPGT = t.TypeVar('EXPGT', bound='CerebrumBaseModel')


# pydantic-based BaseModel for Query and Transform paramters
class BaseParams(CerebrumBaseModel, t.Generic[INPGT, EXPGT]):
    init_params: t.Annotated[INPGT, Field(title="Init. Params")]
    exec_params: t.Annotated[EXPGT, Field(title="Exec. Params")]

   
    @override
    def is_valid_field(self, field: str) -> bool:
        return ((field in self.init_params.model_fields) or
                (field in self.exec_params.model_fields) or 
                (field in self.model_fields))

    @override
    def get(self, field: str, val_not_found: t.Any = None) -> t.Any:
        ivalue = self.init_params.get(field)
        if ivalue is not None:
            return ivalue 
        evalue = self.exec_params.get(field)
        if evalue is not None:
            return evalue 
        return super().get(field, val_not_found)


# pydantic-based model for no paramters
class NoneParams(CerebrumBaseModel):
    pass


# Abstract Base class for parameter interface
class ParamsInterface(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def params_type(
        cls
    ) -> type[BaseParams[INPGT, EXPGT]]:  # pyright:ignore[reportInvalidTypeVarUse]
        return BaseParams

    @classmethod
    @abc.abstractmethod
    def params_instance(
        cls,
        param_dict: dict[str, t.Any],
    ) -> BaseParams[INPGT, EXPGT]:  # pyright:ignore[reportInvalidTypeVarUse]
        return BaseParams.model_validate(param_dict)


XformElt: t.TypeAlias = dict[str, t.Any]
XformItr : t.TypeAlias = Iterable[XformElt] 
XformSeq: t.TypeAlias = Sequence[XformElt]
QryElt: t.TypeAlias = dict[str, t.Any]
QryItr : t.TypeAlias = Iterable[QryElt]


# Abstract interface for Database Queries
class DbQuery(ParamsInterface, abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        in_iter: QryItr | None,
        **params: t.Any,
    ) -> QryItr | None:
        return None


# Abstract interface for XFormer operations
class OpXFormer(ParamsInterface, abc.ABC):
    @abc.abstractmethod
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        return None


# Abstract interface for DBWrite operations
class QryDBWriter(abc.ABC):
    @abc.abstractmethod
    def write(
        self,
        in_iter: QryItr | None,
        **_params: t.Any,
    ) -> None:
        pass


# Abstract Base class for network structure components
class BaseStruct(CerebrumBaseModel, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    @override
    def exclude(self) -> set[str]:
        return set([])

    @abc.abstractmethod
    def apply_mod(self, mod_struct: Self) -> Self:
        return self
