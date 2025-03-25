import abc
import typing as t
from typing_extensions import Self
from pydantic import BaseModel, Field

from collections.abc import Iterable, Sequence

XformElt: t.TypeAlias = dict[str, t.Any]
XformItr : t.TypeAlias = Iterable[XformElt] 
XformSeq: t.TypeAlias = Sequence[XformElt]
QryElt: t.TypeAlias = dict[str, t.Any]
QryItr : t.TypeAlias = Iterable[QryElt]


#
# pydantic base models
#
class CerebrumBaseModel(BaseModel):
    name : t.Annotated[str, Field(title="Name")] = ""

    def get(self, field: str) -> t.Any:
        try:
            return getattr(self, field)
        except AttributeError:
            return None


class BaseParams(CerebrumBaseModel):
    pass



class BaseStruct(CerebrumBaseModel, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def exclude(self) -> set[str]:
        return set([])

    @abc.abstractmethod
    def apply_mod(self, mod_struct: Self) -> Self:
        return self


#
# Abstract Base classes
#
class ParamsInterface(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def params_type(cls) -> type[BaseParams]:
        return BaseParams

    @classmethod
    @abc.abstractmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> BaseParams:
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
