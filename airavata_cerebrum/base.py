import abc
import typing as t
import traitlets

from collections.abc import Iterable, Sequence

XformElt: t.TypeAlias = dict[str, t.Any]
XformItr : t.TypeAlias = Iterable[XformElt] 
XformSeq: t.TypeAlias = Sequence[XformElt]
QryElt: t.TypeAlias = dict[str, t.Any]
QryItr : t.TypeAlias = Iterable[QryElt]
#
# Abstract Base classes
#
class EmptyTraits(traitlets.HasTraits):
    pass


class TraitInterface(abc.ABC):
    @abc.abstractmethod
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return EmptyTraits

    @abc.abstractmethod
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return EmptyTraits(**trait_values)


# Abstract interface for Database Queries
class DbQuery(TraitInterface, abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        in_iter: QryItr | None,
        **params: t.Any,
    ) -> QryItr | None:
        return None


# Abstract interface for XFormer operations
class OpXFormer(TraitInterface, abc.ABC):
    @abc.abstractmethod
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        return None
