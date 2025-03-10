import abc
import itertools
import logging
import typing as t

import marimo as mo
import awitree

from collections.abc import Iterable
import traitlets
from typing_extensions import override
from anywidget import AnyWidget
from marimo._plugins.ui._core.ui_element import UIElement

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
            # _log().warning("options %s",str(kwargs["options"]))
            return mo.ui.dropdown(
                options=kwargs["options"],
                value=kwargs["default"],
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

    def update(self):
        # NOTE: Unlink here
        # NOTE: Create links and append to the 'links' here
        pass


class RecipeSidePanel(PanelBase):
    def __init__(
        self,
        template_map: dict[str, t.Any],
        elt_traits: traitlets.HasTraits | None = None,
        **kwargs: t.Any
    ):
        super().__init__(**kwargs)
        # Setup Widgets
        self.widget_map : dict[str, UIElement[t.Any, t.Any] | None] = (
            self.init_widget_map(template_map, elt_traits)
        )
        # Setup the Layout
        self.layout : mo.Html | None = mo.vstack([
            self.params_widget(
                template_map,
                RecipeKeys.INIT_PARAMS,
                RecipeLabels.INIT_PARAMS,
            ),
            self.params_widget(
                template_map,
                RecipeKeys.EXEC_PARAMS,
                RecipeLabels.EXEC_PARAMS,
            ),
        ])

    def params_widget(
        self,
        template_map: dict[str, t.Any],
        params_key: str,
        params_label: str
    ):
        if template_map[params_key]:
            wd_itr : Iterable[UIElement[t.Any, t.Any] | None] = (
                self.widget_map[pkx] 
                for pkx in template_map[params_key].keys()
            ) 
            return mo.ui.array(
                [wx for wx in wd_itr if wx is not None],
                label=params_label
            )
        else:
            return mo.ui.array(
                [],
                label=params_label + RecipeLabels.NA
            )

    def init_widget_map(
        self,
        template_map: dict[str, t.Any],
        elt_traits: traitlets.HasTraits | None = None,
    ) -> dict[str, UIElement[t.Any, t.Any] | None]:
        widget_map = {}
        traitv_dct = elt_traits.trait_values() if elt_traits else {}
        for ekey, vmap in template_map[RecipeKeys.INIT_PARAMS].items():
            if ekey in traitv_dct:
                vmap["default"] = traitv_dct[ekey]
            widget_map[ekey] = render_property(
                vmap[RecipeKeys.TYPE],
                **vmap
            )
        for ekey, vmap in template_map[RecipeKeys.EXEC_PARAMS].items():
            if ekey in traitv_dct:
                vmap["default"] = traitv_dct[ekey]
            widget_map[ekey] = render_property(
                vmap[RecipeKeys.TYPE],
                **vmap
            )
        return widget_map


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
        self.name: str = name
        self.payload : PayLoad | None = payload
        self.children : dict[str, "CBTreeNode"] = {}
        if nodes:
            self.children = {nx.payload.node_key : nx for nx in nodes}

    def add_node(self, child: "CBTreeNode"):
        self.children[child.payload.node_key] = child

    def awi_dict(self) -> dict[str, t.Any]:
        return {
            'id' : self.payload.node_key,
            'text': self.name,
            'state': {'open': True},
            'children': [
                cx.awi_dict() for _kx, cx in self.children.items()
            ]
        }

    @classmethod
    def init(cls, name: str, key: str) -> "CBTreeNode":
        return cls(name=name, payload=PayLoad(node_key=key))

    @classmethod
    def from_struct(
        cls,
        struct_obj: structure.StructBase,
        node_key: str | None = None,
    ) -> "CBTreeNode":
        return cls(
            name=struct_obj.name,
            payload=struct_payload(struct_obj, node_key)
        )

    @classmethod
    def from_recipe_step(
        cls,
        wf_step: dict[str, t.Any],
        node_key: str | None = None
    ) -> "CBTreeNode":
        return cls(
            name=wf_step[RecipeKeys.LABEL],
            payload=recipe_step_payload(wf_step, node_key)
        )


class TreeBase(abc.ABC):
    def __init__(
        self,
        left_width: float = 0.5,
        **kwargs: t.Any,
    ):
        self.tree: AnyWidget | None = None
        self.layout: mo.Html | None = None
        self.panel_dict: dict[str, PanelBase] = {}
        self.left_width: float = left_width
        self.widths : list[float] = [left_width, 1 - left_width]

    @abc.abstractmethod
    def build(self) -> "TreeBase":
        pass

    def tree_update(self, change: dict[str, t.Any]):
        new_val = change["new"]
        if not len(new_val):
            return
        selected_nodes : list[dict[str, t.Any]] = new_val
        _log().warning(
            "Tree Update selected_nodes : [%s %s]",
            str(change),
            str(selected_nodes)
        )
        selected_id: str = selected_nodes[0]["id"]
        _log().warning(
            "Tree Update ID : [%s]",
            selected_id,
        )
        if selected_id in self.panel_dict:
            side_panel = self.panel_dict[selected_id]
            side_panel.update()
            self.layout = mo.hstack([self.tree, side_panel.layout])


class RecipeTreeBase(TreeBase, metaclass=abc.ABCMeta):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: float=0.4,
        **kwargs: t.Any,
    ):
        super().__init__(left_width, **kwargs)
        self.mdr_setup: RecipeSetup = mdr_setup
        self.root_node: CBTreeNode | None = None

    def db_recipe_node(
        self,
        db_key: str,
        db_desc: dict[str, t.Any]
    ) -> CBTreeNode:
        db_node = CBTreeNode.init(
            name=db_desc[RecipeKeys.LABEL],
            key=db_key
        )
        wflow_desc = db_desc[RecipeKeys.WORKFLOW]
        _log().info(
            "Start Left-side panel construction for [%s]",
            str(db_key)
        )
        for wf_idx, wf_step in enumerate(wflow_desc):
            step_key = wf_step[RecipeKeys.NAME]
            _log().debug("Initializing Panels for [%s]", step_key)
            ptemplate = self.mdr_setup.get_template_for(step_key)
            pkey = f"{db_key}-{wf_idx}-{step_key}"
            rcp_node = CBTreeNode.from_recipe_step(wf_step, pkey)
            self.panel_dict[pkey] = RecipeSidePanel(
                ptemplate,
                rcp_node.payload.node_traits,
            )
            db_node.add_node(rcp_node)
        _log().info("Completed Left-side panel construction")
        return db_node

    @override
    def build(self) -> TreeBase:
        self.tree : AnyWidget | None = self.build_tree()
        # self.tree.observe(self.tree_update, names="selected_nodes")
        self.layout : mo.Html | None = mo.hstack(
            [self.tree, mo.vstack([])],
            widths=[self.left_width, 1-self.left_width])
        return self

    @override
    @abc.abstractmethod
    def build_tree(self) -> AnyWidget:
        pass


class SourceDataTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: float=0.4,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.src_data_desc : dict[str, t.Any] = (
            mdr_setup.recipe_sections[RecipeKeys.SRC_DATA]
        )

    @override
    def build_tree(self) -> AnyWidget:
        self.root_node : CBTreeNode | None = CBTreeNode.init(
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
            self.root_node.add_node(db_node)
        # initialize tree
        tree : AnyWidget = awitree.Tree(
            data=self.root_node.awi_dict()
        )
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree


class D2MLocationsTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: float=0.4,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.d2m_map_desc : dict[str, t.Any] = (
            mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]
        )

    def neuron_node(
        self,
        location_name: str,
        neuron_name: str,
        neuron_desc: dict[str, t.Any]
    ) -> CBTreeNode:
        neuron_node = CBTreeNode.init(
            name=neuron_name,
            key=f"d2m_map-{location_name}-{neuron_name}"
        )
        for db_key, db_desc in neuron_desc[RecipeKeys.SRC_DATA].items():
            db_node = self.db_recipe_node(
                f"{location_name}-{neuron_name}-{db_key}",
                db_desc
            )
            neuron_node.add_node(db_node)
        return neuron_node

    @override
    def build_tree(self) -> AnyWidget:
        self.root_node : CBTreeNode | None = CBTreeNode.init(
            name=RecipeKeys.LOCATIONS,
            key="d2m_map.locations.root"
        )
        for loc_name, loc_desc in self.d2m_map_desc[RecipeKeys.LOCATIONS].items():
            loc_node = CBTreeNode.init(
                name=loc_name,
                key=f"d2m_map-{loc_name}"
            )
            for neuron_name, neuron_desc in loc_desc.items():
                neuron_node = self.neuron_node(
                    loc_name,
                    neuron_name,
                    neuron_desc
                )
                loc_node.add_node(neuron_node)
            self.root_node.add_node(loc_node)
        #TODO: initialize tree
        # initialize tree
        tree : AnyWidget = awitree.Tree(
            data=self.root_node.awi_dict()
        )
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree
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
