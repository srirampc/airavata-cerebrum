import abc
import itertools
import logging
import types
import typing as t

import marimo as mo
import traitlets

from collections.abc import Iterable
from typing_extensions import override
from anywidget import AnyWidget
from marimo._plugins.ui._core.ui_element import UIElement

from ..base import DbQuery, OpXFormer
from ..register import find_type
from ..model.setup import RecipeKeys, RecipeLabels, RecipeSetup
from ..model import structure as structure
from . import PayLoad, RcpTreeNames, recipe_step_payload, struct_payload

def _log():
    return logging.getLogger(__name__)


def scalar_widget(
    widget_key: str,
    label: str = "",
    **kwargs: t.Any
) -> UIElement[t.Any, t.Any] | None:
    match widget_key:
        case "int" | "int32" | "int64":
            return mo.ui.number(
                value=kwargs["default"],
                label=label
            )
        case "float" | "float32" | "float64":
            return mo.ui.number(
                value=kwargs["default"],
                label=label,
            )
        case "text":
            return mo.ui.text(
                value=kwargs["default"],
                disabled=False,
                label=label,
            )
        case "textarea" | "str":
            return mo.ui.text_area(
                value=kwargs["default"],
                disabled=False,
                label=label,
            )
        case "option":
            return mo.ui.dropdown(
                options=kwargs["options"],
                label=label,
            )
        case "check" | "bool":
            return mo.ui.checkbox(
                value=bool(kwargs["default"]),
                label=label,
            )
        case "tags":
            return mo.ui.multiselect(
                options=kwargs["allowed"],
                value=kwargs["default"],
                label=label,
            )
        case _:
            return None


# NOTE: Features of trait setting and change handler not included 
class PropertyListLayout(mo.ui.array):
    def __init__(
        self,
        value: list[t.Any],
        label:str="",
        **kwargs: t.Any
    ):
        super().__init__(
            [wx for wx in self.widgets(value) if wx],
            label=label,
            **kwargs
        )

    def widgets(self, value: list[t.Any]):
        return(
            scalar_widget(
                type(vx).__name__,
                label=str(kx) + " :",
                default=vx
            ) for kx, vx in enumerate(value) 
        )


# NOTE: Features of trait setting and change handler not included 
class PropertyMapLayout(mo.ui.dictionary):

    def __init__(self, value: dict[str, t.Any], **kwargs: t.Any):
        super().__init__(
            {kx: wx for kx, wx in self.widgets(value) if wx},
            **kwargs
        )
        # print("Value: ", len(value))

    def init_widget(
        self,
        vx: t.Any,
        label:str = ""
    ) -> UIElement[t.Any, t.Any] | None:
        tname = type(vx).__name__
        match tname:
            case "NoneType":
                return scalar_widget("str", default="None", label=label)
            case "list":
                return PropertyListLayout(vx, label="")
            case _:
                return scalar_widget(tname, default=vx, label=label)

    def widgets(
        self,
        value: dict[str, t.Any]
    ) -> Iterable[tuple[str, UIElement[t.Any, t.Any] | None]]:
        return (
            (kx, self.init_widget(vx)) for kx, vx in value.items()
        )

def render_property(
    widget_key: str,
    label:str = "",
    **kwargs: t.Any
) -> UIElement[t.Any, t.Any] | None:
    match widget_key:
        case "dict":
            return PropertyMapLayout(value=kwargs["default"], **kwargs)
        case "list":
            return PropertyListLayout(value=kwargs["default"], **kwargs)
        case _:
            return scalar_widget(widget_key, label=label, **kwargs)


# Base class for side panel
# NOTE: Features of update and clearing links not included 
class PanelBase:
    def __init__(self, **kwargs: t.Any):
        self.layout : mo.Html | None = None
        self.widget_map : dict[str, UIElement[t.Any, t.Any] | None] = {}
        self.links : list[tuple[t.Any, t.Any]] = []

    def clear_links(self):
        # NOTE: Unlink here
        self.links.clear()

    def update(self, new_val: t.Any):
        self.clear_links()
        if not new_val:
            return
        # NOTE: Create links and append to the 'links' here


class RecipeSidePanel(PanelBase):
    def __init__(self, template_map: dict[str, t.Any], **kwargs: t.Any):
        super().__init__(**kwargs)
        for ekey, vmap in template_map[RecipeKeys.INIT_PARAMS].items():
            self.widget_map[ekey] = render_property(
                vmap[RecipeKeys.TYPE],
                **vmap
            )
        for ekey, vmap in template_map[RecipeKeys.EXEC_PARAMS].items():
            self.widget_map[ekey] = render_property(
                vmap[RecipeKeys.TYPE],
                **vmap
            )

        # Set Widgets
        ip_widgets: mo.ui.array = mo.ui.array([])
        if (
            RecipeKeys.INIT_PARAMS in template_map
            and template_map[RecipeKeys.INIT_PARAMS]
        ):
            wd_itr : Iterable[UIElement[t.Any, t.Any] | None] = (
                self.widget_map[pkx] 
                for pkx in template_map[RecipeKeys.INIT_PARAMS].keys()
            ) 
            ip_widgets = mo.ui.array(
                [wx for wx in wd_itr if wx is not None],
                label=RecipeLabels.INIT_PARAMS
            )
        else:
            ip_widgets = mo.ui.array(
                [],
                label=RecipeLabels.INIT_PARAMS + RecipeLabels.NA
            )
        ep_widgets: mo.ui.array = mo.ui.array([])
        if (
            RecipeKeys.EXEC_PARAMS in template_map
            and template_map[RecipeKeys.EXEC_PARAMS]
        ):
            ewd_itr : Iterable[UIElement[t.Any, t.Any] | None] = (
                self.widget_map[pkx] 
                for pkx in template_map[RecipeKeys.EXEC_PARAMS].keys()
            ) 
            ep_widgets = mo.ui.array(
                [wx for wx in ewd_itr if wx is not None],
                label=RecipeLabels.EXEC_PARAMS
            )
        else:
            ep_widgets = mo.ui.array(
                [],
                label=RecipeLabels.EXEC_PARAMS + RecipeLabels.NA
            )
        self.layout : mo.Html | None = mo.vstack([ip_widgets, ep_widgets])


class StructSidePanel(PanelBase):
    def __init__(
        self,
        struct_comp: structure.StructBase,
        **kwargs: t.Any
    ):
        super().__init__(**kwargs)
        str_dict = struct_comp.model_dump()
        for ekey, vmap in struct_comp.trait_ui().items():
            self.widget_map[ekey] = render_property(
                vmap.value_type,
                label=vmap.label  + " :",
                default=vmap.to_ui(str_dict[ekey]),
            )

        # Set Widgets
        wd_itr : Iterable[UIElement[t.Any, t.Any] | None] = (
            self.widget_map[kx] for kx in struct_comp.trait_ui().keys()
        )
        struct_widgets = mo.ui.array(
            [wx for  wx in wd_itr if wx],
            label=struct_comp.name
        )
        self.layout : mo.Html | None = mo.vstack([struct_widgets])


#
# Base class for tree node
class CBTreeNode:
    def __init__(
        self,
        name: str,
        payload: PayLoad | None,
        nodes: list["CBTreeNode"] | None= None,
        **kwargs : t.Any
    ):
        self.payload : PayLoad | None = payload
        self.children : dict[str, "CBTreeNode"] = {}
        if nodes:
            self.children = {nx.payload.node_key : nx for nx in nodes}

    def add_node(self, child: "CBTreeNode"):
        self.children[child.payload.node_key] = child

    @classmethod
    def init(cls, name: str, key: str) -> "CBTreeNode":
        return cls(name=name, payload=PayLoad(node_key=key))

    @classmethod
    def from_struct(cls, struct_obj: structure.StructBase) -> "CBTreeNode":
        return cls(
            name=struct_obj.name,
            payload=struct_payload(struct_obj)
        )
    @classmethod
    def from_recipe_step(cls, wf_step: dict[str, t.Any]) -> "CBTreeNode":
        return cls(
            name=wf_step[RecipeKeys.LABEL],
            payload=recipe_step_payload(wf_step)
        )


class TreeBase(abc.ABC):
    def __init__(
        self,
        left_width: str,
        **kwargs: t.Any,
    ):
        self.tree: AnyWidget | None = None
        self.layout: mo.Html | None = None
        self.panel_keys: set[str] = set([])
        self.panel_dict: dict[str, PanelBase] = {}
        self.left_width: str = left_width

    @abc.abstractmethod
    def build(self) -> "TreeBase | None":
        pass

    def tree_update(self, change: dict[str, t.Any]):
        new_val = change["new"]
        if not len(new_val):
            return
        pload : PayLoad = new_val[0].payload 
        node_key = pload.node_key
        node_traits = pload.node_traits
        _log().warning(
            "Tree Update : [%s %s %s]",
            node_key,
            str(node_key in self.panel_dict),
            str(node_traits.trait_names() if node_traits else None)
        )
        if node_key in self.panel_dict:
            side_panel = self.panel_dict[node_key]
            side_panel.update(node_traits)
            self.layout = mo.hstack([self.tree, side_panel.layout])


class RecipeTreeBase(TreeBase, metaclass=abc.ABCMeta):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str,
        **kwargs: t.Any,
    ):
        super().__init__(left_width, **kwargs)
        self.mdr_setup: RecipeSetup = mdr_setup

    def db_recipe_node(
        self,
        db_key: str,
        db_desc: dict[str, t.Any]
    ) -> CBTreeNode:
        db_node = CBTreeNode.init(
            name=db_desc[RecipeKeys.LABEL],
            key=db_key
        )
        for wf_step in db_desc[RecipeKeys.WORKFLOW]:
            db_node.add_node(CBTreeNode.from_recipe_step(wf_step))
            self.panel_keys.add(wf_step[RecipeKeys.NAME])
        return db_node

    def build_side_panels(self) -> dict[str, PanelBase]:
        _log().info(
            "Start Left-side panel construction for [%s]",
            str(self.panel_dict.keys())
        )
        for pkey in self.panel_keys:
            _log().debug("Initializing Panels for [%s]", pkey)
            ptemplate = self.mdr_setup.get_template_for(pkey)
            self.panel_dict[pkey] = RecipeSidePanel(ptemplate)
        _log().info("Completed Left-side panel construction")
        return self.panel_dict

    @override
    def build(self) -> TreeBase:
        self.tree : AnyWidget | None = self.build_tree()
        self.panel_dict : dict[str, PanelBase]= self.build_side_panels()
        self.tree.observe(self.tree_update, names="selected_nodes")
        self.layout : mo.Html | None = mo.hstack([self.tree, mo.vstack([])])
        return self

    @override
    @abc.abstractmethod
    def build_tree(self) -> AnyWidget:
        pass


class SourceDataTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.src_data_desc : dict[str, t.Any] = (
            mdr_setup.recipe_sections[RecipeKeys.SRC_DATA]
        )

    @override
    def build_tree(self) -> AnyWidget:
        root_node = CBTreeNode.init(
            name=RcpTreeNames.SRC_DATA,
            key="source_data"
        )
        for db_key, db_desc in self.src_data_desc.items():
            db_node = self.db_recipe_node(
                db_key,
                {
                    RecipeKeys.LABEL: db_desc[RecipeKeys.LABEL],
                    RecipeKeys.WORKFLOW: itertools.chain(
                        db_desc[RecipeKeys.DB_CONNECT][RecipeKeys.WORKFLOW],
                        db_desc[RecipeKeys.POST_OPS][RecipeKeys.WORKFLOW],
                    ),
                },
            )
            root_node.add_node(db_node)
        #TODO: initialize tree
        self.tree : itree.Tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type:ignore
        return self.tree


# class D2MLocationsTreeView(RecipeTreeBase):
#     def __init__(
#         self,
#         mdr_setup: RecipeSetup,
#         left_width: str="40%",
#         **kwargs: t.Any,
#     ) -> None:
#         super().__init__(mdr_setup, left_width, **kwargs)
#         self.d2m_map_desc : dict[str, t.Any] = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]
# 
#     def init_neuron_node(
#         self, neuron_name: str, neuron_desc: dict[str, t.Any]
#     ) -> CBTreeNode:
#         neuron_node = CBTreeNode(name=neuron_name, node_key="d2m_map.neuron")
#         for db_key, db_desc in neuron_desc[RecipeKeys.SRC_DATA].items():
#             neuron_node.add_node(self.recipe_step_node(db_key, db_desc))
#         return neuron_node
# 
#     @override
#     def init_tree(self) -> AnyWidget:
#         root_node = CBTreeNode(name=RecipeKeys.LOCATIONS, node_key="root")
#         for loc_name, loc_desc in self.d2m_map_desc[RecipeKeys.LOCATIONS].items():
#             loc_node = CBTreeNode(name=loc_name, node_key="d2m_map.location")
#             for neuron_name, neuron_desc in loc_desc.items():
#                 neuron_node = self.init_neuron_node(neuron_name, neuron_desc)
#                 loc_node.add_node(neuron_node)
#             root_node.add_node(loc_node)
#         #TODO: initialize tree
#         self.tree : itree.Tree = itree.Tree(multiple_selection=False)
#         self.tree.add_node(root_node)
#         self.tree.layout.width = self.left_width  # type: ignore
#         return self.tree
# 
# 
# class D2MConnectionsTreeView(RecipeTreeBase):
#     def __init__(
#         self,
#         mdr_setup: RecipeSetup,
#         left_width: str="40%",
#         **kwargs: t.Any,
#     ) -> None:
#         super().__init__(mdr_setup, left_width, **kwargs)
#         self.d2m_map_desc : dict[str, t.Any] = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]
# 
#     def init_conncection_node(
#         self, connect_name: str, connect_desc: dict[str, t.Any]
#     ) -> CBTreeNode:
#         conn_node = CBTreeNode(name=connect_name, node_key="d2m_map.connection")
#         for db_key, db_desc in connect_desc[RecipeKeys.SRC_DATA].items():
#             db_node = CBTreeNode(name=db_desc[RecipeKeys.LABEL], node_key=db_key)
#             for wf_step in db_desc[RecipeKeys.WORKFLOW]:
#                 db_node.add_node(RecipeTreeBase.wflow_step_tree_node(wf_step))
#             self.panel_keys.union(
#                 set(
#                     wf_step[RecipeKeys.NAME] for wf_step in db_desc[RecipeKeys.WORKFLOW]
#                 )
#             )
#             conn_node.add_node(self.recipe_step_node(db_key, db_desc))
#         return conn_node
# 
#     @override
#     def init_tree(self) -> AnyWidget:
#         root_node = CBTreeNode(name=RcpTreeNames.CONNECTIONS, node_key="tree")
#         for conn_name, conn_desc in self.d2m_map_desc[RecipeKeys.CONNECTIONS].items():
#             root_node.add_node(self.init_conncection_node(conn_name, conn_desc))
#         #TODO: initialize tree
#         self.tree : itree.Tree = itree.Tree(multiple_selection=False)
#         self.tree.add_node(root_node)
#         self.tree.layout.width = self.left_width  # type: ignore
#         return self.tree
