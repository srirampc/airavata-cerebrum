import abc
import typing as t
#
from collections.abc import Iterable, Sequence
from typing_extensions import Self, override
from pydantic import BaseModel, Field


XformElt: t.TypeAlias = dict[str, t.Any]
XformItr : t.TypeAlias = Iterable[XformElt] 
XformSeq: t.TypeAlias = Sequence[XformElt]
QryElt: t.TypeAlias = dict[str, t.Any]
QryItr : t.TypeAlias = Iterable[QryElt]


#
# pydantic-based BaseModel for Cerebrum
class CerebrumBaseModel(BaseModel):
    name : t.Annotated[str, Field(title="Name")] = ""

    def get(self, field: str) -> t.Any:
        try:
            return getattr(self, field)
        except AttributeError:
            return None


INPGT = t.TypeVar('INPGT', bound='CerebrumBaseModel')
EXPGT = t.TypeVar('EXPGT', bound='CerebrumBaseModel')


class BaseParams(CerebrumBaseModel, t.Generic[INPGT, EXPGT]):
    init_params: t.Annotated[INPGT, Field(title="Init. Params")]
    exec_params: t.Annotated[EXPGT, Field(title="Exec. Params")]

    @override
    def get(self, field: str) -> t.Any:
        ivalue = self.init_params.get(field)
        if ivalue is not None:
            return ivalue 
        evalue = self.exec_params.get(field)
        if evalue is not None:
            return evalue 
        return super().get(field)


# Abstract Base class for network structure components
class BaseStruct(CerebrumBaseModel, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def exclude(self) -> set[str]:
        return set([])

    @abc.abstractmethod
    def apply_mod(self, mod_struct: Self) -> Self:
        return self


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
