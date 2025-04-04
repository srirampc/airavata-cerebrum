import logging
import abc
import typing as t
import awitree
#
from typing_extensions import Self

#
from ..base import BaseStruct, BaseParams, CerebrumBaseModel
from ..base import OpXFormer, DbQuery
from ..base import InitParamsT, ExecParamsT
from ..model.setup import RecipeKeys
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


def workflow_params_dict(
    wf_step: dict[str, t.Any],
) -> dict[str, t.Any]:
    return (
        {
            RecipeKeys.NAME: wf_step[RecipeKeys.LABEL],
        }
        | wf_step[RecipeKeys.INIT_PARAMS]
        | wf_step[RecipeKeys.EXEC_PARAMS]
    )

def workflow_params(
    wf_step: dict[str, t.Any],
    node_key: str | None = None,
) -> tuple[str | None,  BaseParams[CerebrumBaseModel, CerebrumBaseModel] | None]:

    node_key = node_key if node_key else wf_step[RecipeKeys.NAME]
    src_class: type[DbQuery] | type[OpXFormer] | None = find_type(
        wf_step[RecipeKeys.NAME]
    )
    if src_class:
        return node_key, src_class.params_instance(
            wf_step | 
            {
                RecipeKeys.NAME: wf_step[RecipeKeys.LABEL],
            }
        )
    else:
        return node_key, None


class PayLoad:
    def __init__(
        self,
        node_key: str,
        node_props: CerebrumBaseModel  | None = None
    ) -> None:
        self.node_key: str = node_key
        self.node_props: CerebrumBaseModel | None = node_props

    @classmethod
    def from_struct(
        cls,
        struct_obj: BaseStruct,
        node_key: str | None = None,
    )-> Self:
        return cls(
            node_key=node_key if node_key else struct_obj.name,
        )
  
    @classmethod
    def from_recipe_step(
        cls,
        wf_step: dict[str, t.Any],
        node_key: str | None = None,
    ) -> Self | None:
        node_key,  wf_params = workflow_params(wf_step, node_key)
        if node_key:
            if wf_params:
                plx = cls(
                    node_key=node_key,
                    node_props=wf_params,
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
                    str(wf_params),
                    str(wf_params)
                )
                return cls(node_key, **workflow_params_dict(wf_step))
        else:
            return None


# Base class for side panel
# NOTE: Features of update and clearing links not included
LayoutType = t.TypeVar('LT')
UIEltType = t.TypeVar('UT')
class BasePanel(t.Generic[LayoutType, UIEltType]):
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


PanelType = t.TypeVar('PBT', bound=BasePanel[LayoutType, UIEltType])


class BaseTree(abc.ABC, t.Generic[PanelType]):
    def __init__(
        self,
        **kwargs: t.Any,
    ):
        self.tree: awitree.Tree | None = None
        self.layout_: PanelType.LT | None = None
        self.panel_dict: dict[str, PanelType] = {}

    @abc.abstractmethod
    def build(self, root_pfx: str = "") -> "BaseTree[PanelType]":
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
        struct_obj: BaseStruct,
        node_key: str | None = None,
    ) -> Self:
        return cls(
            name=struct_obj.name,
            payload=PayLoad.from_struct(struct_obj, node_key)
        )

    @classmethod
    def from_recipe_step(
        cls,
        wf_step: dict[str, t.Any],
        node_key: str | None = None,
    ) -> Self:
        return cls(
            name=wf_step[RecipeKeys.LABEL],
            payload=PayLoad.from_recipe_step(wf_step, node_key),
        )
