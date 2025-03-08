import typing as t

import traitlets

from ..base import DbQuery, OpXFormer
from ..model.setup import RecipeKeys
from ..model.structure import StructBase
from ..register import find_type


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



class TNode(traitlets.HasTraits):
    node_key: traitlets.Unicode[str, str | bytes] = traitlets.Unicode()
    node_traits: traitlets.HasTraits = traitlets.HasTraits()


def struct_tnode(struct_obj: StructBase) -> TNode:
    return TNode(
        node_key=struct_obj.name,
        node_trait=struct_obj.trait_instance(
            **struct_obj.model_dump(exclude=struct_obj.exclude()),
        ),
    )


def recipe_step_tnode(wf_step: dict[str, t.Any]) -> TNode | None:
    wf_dict = (
        {
            RecipeKeys.NAME: wf_step[RecipeKeys.LABEL],
            RecipeKeys.NODE_KEY: wf_step[RecipeKeys.NAME],
        }
        | wf_step[RecipeKeys.INIT_PARAMS]
        | wf_step[RecipeKeys.EXEC_PARAMS]
    )
    src_class: type[DbQuery] | type[OpXFormer] | None = find_type(
        wf_step[RecipeKeys.NAME]
    )
    if RecipeKeys.NODE_KEY in wf_dict:
        if src_class:
            return TNode(
                node_key=wf_dict[RecipeKeys.NODE_KEY],
                node_traits=src_class.trait_instance(**wf_dict),
            )
        else:
            return TNode(**wf_dict)
    else:
        return None
