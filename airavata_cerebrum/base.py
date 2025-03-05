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


# Abstract interface for Database Queries
class DbQuery(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        **params: t.Any
    ):
        return None

    @abc.abstractmethod
    def run(
        self,
        in_iter: QryItr | None,
        **params: t.Any,
    ) -> QryItr | None:
        return None

    @classmethod
    @abc.abstractmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return EmptyTraits


# Abstract interface for XFormer operations
class OpXFormer(abc.ABC):
    @abc.abstractmethod
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        return None

    @classmethod
    @abc.abstractmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return EmptyTraits
