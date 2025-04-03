import itertools
import typing as t
from collections.abc import Iterable

from .base import OpXFormer, DbQuery, QryDBWriter
from .dataset import abc_mouse as abc_mouse_db
from .dataset import abm_celltypes as abm_celltypes_db
from .dataset import ai_synphys as ai_synphys_db
from .dataset import me_features as me_features_db
from .operations import abc_mouse as abc_mouse_ops
from .operations import abm_celltypes as abm_celltypes_ops
from .operations import ai_synphys as ai_synphys_ops
from .operations import json_filter
from .operations import dict_filter
from .operations import xform
from .util import class_qual_name


#
# -------- Register Query and Xform classes
#
RCType = t.TypeVar("RCType")


class TypeRegister(t.Generic[RCType]):
    register_map: dict[str, type[RCType]]

    def __init__(
        self,
        register_lst: Iterable[type[RCType]],
        base_class: type[RCType],
        key_source: t.Literal['class', 'module'] = 'class'
    ) -> None:
        self.register_map = {
            self.qual_name(clsx, key_source): clsx
            for clsx in register_lst
            if issubclass(clsx, base_class)
        }
    
    def qual_name(
        self,
        clsx: type,
        key_source: t.Literal['class', 'module'],
    ) -> str:
        match key_source:
            case 'class':
                return class_qual_name(clsx)
            case 'module':
                return clsx.__module__

    def get_object(
        self, query_key: str, **init_params: t.Any
    ) -> RCType | None:
        if query_key not in self.register_map:
            return None
        return self.register_map[query_key](**init_params)

    def get_type(self, query_key: str) -> type[RCType] | None:
        if query_key not in self.register_map:
            return None
        return self.register_map[query_key]


QUERY_REGISTER: TypeRegister[DbQuery] = TypeRegister(
    itertools.chain(
        abc_mouse_db.query_register(),
        abm_celltypes_db.query_register(),
        ai_synphys_db.query_register(),
        me_features_db.quer_register(),
        xform.query_register(),
        json_filter.query_register(),
        dict_filter.query_register(),
        abc_mouse_ops.query_register(),
        abm_celltypes_ops.query_register(),
        ai_synphys_ops.query_register(),
    ),
    DbQuery,
)

XFORM_REGISTER: TypeRegister[OpXFormer] = TypeRegister(
    itertools.chain(
        abc_mouse_db.xform_register(),
        abm_celltypes_db.xform_register(),
        ai_synphys_db.xform_register(),
        me_features_db.xform_register()
        xform.xform_register(),
        json_filter.xform_register(),
        dict_filter.xform_register(),
        abc_mouse_ops.xform_register(),
        abm_celltypes_ops.xform_register(),
        ai_synphys_ops.xform_register(),
    ),
    OpXFormer,
)

QRY_DBWIRTER_REGISTER: TypeRegister[QryDBWriter] = TypeRegister(
    [
        abm_celltypes_db.dbwriter_register(),
        abc_mouse_db.dbwriter_register(),
        ai_synphys_db.dbwriter_register(),
        me_features_db.dbwriter_register(),
    ],
    QryDBWriter,
    key_source='module'
)


def find_type(
    register_key: str,
) -> type[OpXFormer] | type[DbQuery] | None:
    reg_type = QUERY_REGISTER.get_type(register_key)
    if reg_type:
        return reg_type
    return XFORM_REGISTER.get_type(register_key)


def get_query_object(
    register_key: str,
    **params: t.Any,
) -> DbQuery | None:
    return QUERY_REGISTER.get_object(
        register_key, **params
    )


def get_xform_op_object(
    register_key: str,
    **params: t.Any,
) -> OpXFormer | None:
    return XFORM_REGISTER.get_object(
        register_key, **params
    )

def get_query_db_writer_object(
    register_key: str,
    **params: t.Any,
) -> QryDBWriter | None:
    return QRY_DBWIRTER_REGISTER.get_object(
        register_key, **params
    )
