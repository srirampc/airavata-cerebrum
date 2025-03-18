import logging
import abc
import typing as t

import traitlets

import awitree

from ..base import DbQuery, OpXFormer
from ..model.setup import RecipeKeys
from ..model.structure import StructBase
from ..register import find_type

def _log():
    return logging.getLogger(__name__)

@t.final
class RcpTreeNames:
    RECIPE = "Recipe"
    SRC_DATA = "Data Sources"
    LOCATIONS = "Locations"
    CONNECTIONS = "Connections"
    REGIONS = "Regions"
    D2M = "Data->Model"
    D2M_NEURON = "d2m_map.neuron"
    D2M_LOCATION = "d2m_map.location"
    D2M_CONNECION =  "d2m_map.connection"


@t.final
class StructTreeNames:
    SRC_DATA = "Data Sources"
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

def workflow_params(
    wf_step: dict[str, t.Any],
    node_key: str | None = None,
) -> dict[str, t.Any]:
    return (
        {
            RecipeKeys.NAME: wf_step[RecipeKeys.LABEL],
            RecipeKeys.NODE_KEY: (
                node_key if node_key else wf_step[RecipeKeys.NAME]
            ),
        }
        | wf_step[RecipeKeys.INIT_PARAMS]
        | wf_step[RecipeKeys.EXEC_PARAMS]
    )

def recipe_traits(
    wf_step: dict[str, t.Any],
    node_key: str | None = None,
) -> tuple[dict[str, t.Any], traitlets.HasTraits | None]:
    src_class: type[DbQuery] | type[OpXFormer] | None = find_type(
        wf_step[RecipeKeys.NAME]
    )
    wf_dict: dict[str, t.Any] = workflow_params(wf_step, node_key)
    if src_class:
        return wf_dict, src_class.trait_instance(**wf_dict)
    else:
        return wf_dict, None

def recipe_step_payload(
    wf_step: dict[str, t.Any],
    node_key: str | None = None,
) -> PayLoad | None:
    wf_dict,  rcp_traits = recipe_traits(wf_step, node_key)
    if RecipeKeys.NODE_KEY in wf_dict:
        if rcp_traits:
            nkey=wf_dict[RecipeKeys.NODE_KEY]
            plx = PayLoad(
                node_key=nkey,
                node_traits=rcp_traits,
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
                str(rcp_traits)
            )
            return PayLoad(**wf_dict)
    else:
        return None


# Base class for side panel
# NOTE: Features of update and clearing links not included
LayoutType = t.TypeVar('LT')
UIEltType = t.TypeVar('UT')
class PanelBase(t.Generic[LayoutType, UIEltType]):
    def __init__(self, layout: LayoutType | None = None):
        self._layout: LayoutType | None = layout
        self.ui_elements: list[UIEltType] = []

    def update(self):
        # NOTE: Called when selected node is changed
        pass

    @abc.abstractmethod
    def build_layout(self) -> LayoutType | None:
        pass

    @property
    def layout(self) -> LayoutType | None:
        if self._layout is not None:
            return self._layout
        else:
            self.set_layout(self.build_layout())
            return self._layout

    def set_layout(self, value: LayoutType | None):
        self._layout = value


PanelType = t.TypeVar('PBT', PanelBase[LayoutType, UIEltType])
class TreeBase(abc.ABC, t.Generic[PanelType]):
    def __init__(
        self,
        **kwargs: t.Any,
    ):
        self.tree: awitree.Tree | None = None
        self.layout_: PanelType.LT | None = None
        self.panel_dict: dict[str, PanelType] = {}

    @abc.abstractmethod
    def build(self, root_pfx: str = "") -> "TreeBase[PanelType]":
        pass
    
    @abc.abstractmethod
    def set_layout(self, selected_panel: PanelType) -> None:
        pass

    def tree_update(self, change: dict[str, t.Any]):
        new_val = change["new"]
        if not len(new_val):
            return
        selected_nodes: list[dict[str, t.Any]] = new_val
        _log().warning(
            "Tree Update selected_nodes : [%s %s]", str(change), str(selected_nodes)
        )
        selected_id: str = selected_nodes[0]["id"]
        _log().warning(
            "Tree Update ID : [%s]",
            selected_id,
        )
        if selected_id in self.panel_dict:
            side_panel = self.panel_dict[selected_id]
            side_panel.update()
            self.set_layout(side_panel)

    def view_components(
        self,
    ) -> tuple[awitree.Tree | None,
               tuple[dict[str, PanelType], ...]]:
        return (self.tree, (self.panel_dict,))

    @staticmethod
    def panel_selector(
        axtree: awitree.Tree, taxpanel_dict: tuple[dict[str, PanelType], ...]
    ) -> PanelType | None:
        selected_node_id = (
            axtree.selected_nodes[0]["id"]
            if (axtree.selected_nodes and len(axtree.selected_nodes) > 0)
            else None
        )
        if selected_node_id is not None:
            for axp_dict in taxpanel_dict:
                if selected_node_id in axp_dict:
                    return axp_dict[selected_node_id]
        return None

#
# Base class for tree node
class CBTreeNode:
    def __init__(
        self,
        name: str,
        payload: PayLoad | None,
        nodes: list["CBTreeNode"] | None = None,
        **kwargs: t.Any,
    ):
        self.name: str = name
        self.payload: PayLoad | None = payload
        self.children: dict[str, "CBTreeNode"] = {}
        if nodes:
            self.children = {nx.payload.node_key: nx for nx in nodes}

    def add_node(self, child: "CBTreeNode"):
        self.children[child.payload.node_key] = child

    def awi_dict(self) -> dict[str, t.Any]:
        return {
            "id": self.payload.node_key,
            "text": self.name,
            "state": {"disabled": False},
            "children": [cx.awi_dict() for _kx, cx in self.children.items()],
        }

    @classmethod
    def init(cls, name: str, key: str) -> "CBTreeNode":
        return cls(name=name, payload=PayLoad(node_key=key))

    @classmethod
    def from_struct(
        cls,
        struct_obj: StructBase,
        node_key: str | None = None,
    ) -> "CBTreeNode":
        return cls(
            name=struct_obj.name,
            payload=struct_payload(struct_obj, node_key)
        )

    @classmethod
    def from_recipe_step(
        cls, wf_step: dict[str, t.Any], node_key: str | None = None
    ) -> "CBTreeNode":
        return cls(
            name=wf_step[RecipeKeys.LABEL],
            payload=recipe_step_payload(wf_step, node_key),
        )
