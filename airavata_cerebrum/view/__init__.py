import logging
import typing as t

import traitlets

from ..base import DbQuery, OpXFormer
from ..model.setup import RecipeKeys
from ..model.structure import StructBase
from ..register import find_type

def _log():
    return logging.getLogger(__name__)

@t.final
class RcpTreeNames:
    SRC_DATA = "Source Data"
    LOCATIONS = "Locations"
    CONNECTIONS = "Connections"
    REGIONS = "Regions"
    D2M_NEURON = "d2m_map.neuron"
    D2M_LOCATION = "d2m_map.location"
    D2M_CONNECION =  "d2m_map.connection"


@t.final
class StructTreeNames:
    SRC_DATA = "Source Data"
    LOCATIONS = "Locations"
    CONNECTIONS = "Connections"
    REGIONS = "Regions"
    EXTERNAL_NETWORKS = "External Networks"
    DATA_LINKS = "Data Links"
    DATA_FILES = "Data Files"
    D2M_NEURON = "d2m_map.neuron"
    D2M_LOCATION = "d2m_map.location"
    D2M_CONNECION =  "d2m_map.connection"



class PayLoad:
    def __init__(
        self,
        node_key: str,
        node_traits: traitlets.HasTraits | None = None
    ) -> None:
        self.node_key: str = node_key
        self.node_traits: traitlets.HasTraits | None = node_traits


def struct_payload(
    struct_obj: StructBase,
    node_key: str | None = None,
) -> PayLoad:
    return PayLoad(
        node_key=node_key if node_key else struct_obj.name,
        node_traits=struct_obj.trait_instance(
            **struct_obj.model_dump(exclude=struct_obj.exclude()),
        ),
    )


def recipe_step_payload(
    wf_step: dict[str, t.Any],
    node_key: str | None = None,
) -> PayLoad | None:
    wf_dict = (
        {
            RecipeKeys.NAME: wf_step[RecipeKeys.LABEL],
            RecipeKeys.NODE_KEY: (
                node_key if node_key else wf_step[RecipeKeys.NAME]
            ),
        }
        | wf_step[RecipeKeys.INIT_PARAMS]
        | wf_step[RecipeKeys.EXEC_PARAMS]
    )
    src_class: type[DbQuery] | type[OpXFormer] | None = find_type(
        wf_step[RecipeKeys.NAME]
    )
    if RecipeKeys.NODE_KEY in wf_dict:
        if src_class:
            nkey=wf_dict[RecipeKeys.NODE_KEY]
            ntraits = src_class.trait_instance(**wf_dict)
            plx = PayLoad(
                node_key=nkey,
                node_traits=ntraits,
            )
            # _log().warning(
            #     "Data [%s %s %s %s %s]",
            #     str(wf_dict),
            #     str(src_class),
            #     str(nkey),
            #     str(ntraits.trait_names()),
            #     str(plx.node_traits.trait_names()),
            # )
            return plx
        else:
            _log().warning(
                "Default Data [%s %s]",
                str(wf_dict),
                str(src_class)
            )
            return PayLoad(**wf_dict)
    else:
        return None
