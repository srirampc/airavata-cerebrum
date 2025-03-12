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
from . import PayLoad, RcpTreeNames, StructTreeNames, recipe_step_payload, struct_payload

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
            [wx for wx in self.widgets(value, **kwargs) if wx is not None],
            label=label,
        )

    def widgets(self, value: list[t.Any], **_kwargs: t.Any):
        return(
            scalar_widget(
                type(vx).__name__,
                label=str(kx) + " :",
                default=vx
            ) for kx, vx in enumerate(value) 
        )


# NOTE: Features of trait setting and change handler not included 
class PropertyMapLayout(mo.ui.dictionary):

    def __init__(self, value: dict[str, t.Any], label:str="", **kwargs: t.Any):
        super().__init__(
            {kx: wx for kx, wx in self.widgets(value, **kwargs) if wx is not None},
            label=label,
        )
        # print("Value: ", len(value))

    def init_widget(
        self,
        vx: t.Any,
        label:str = "",
         **_kwargs: t.Any
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
        value: dict[str, t.Any],
         **kwargs: t.Any
    ) -> Iterable[tuple[str, UIElement[t.Any, t.Any] | None]]:
        return (
            (kx, self.init_widget(vx, **kwargs)) for kx, vx in value.items()
        )

def render_property(
    widget_key: str,
    label:str = "",
    **kwargs: t.Any
) -> UIElement[t.Any, t.Any] | None:
    # _log().warning("Render Out [%s]", str(kwargs))
    match widget_key:
        case "dict":
            return PropertyMapLayout(value=kwargs["default"], label=label, **kwargs)
        case "list":
            return PropertyListLayout(value=kwargs["default"], **kwargs)
        case _:
            return scalar_widget(widget_key, label=label, **kwargs)


# Base class for side panel
# NOTE: Features of update and clearing links not included 
class PanelBase:
    def __init__(self, layout: mo.Html | None=None):
        self._layout : mo.Html | None = layout
        self.ui_elements : list[UIElement[t.Any, t.Any]] = []

    @property
    def layout(self):
        return self._layout

    def update(self):
        # NOTE: Unlink here
        # NOTE: Create links and append to the 'links' here
        pass


class WorkflowStepPanel(PanelBase):
    def __init__(
        self,
        template_map: dict[str, t.Any],
        add_layout: bool,
        elt_traits: traitlets.HasTraits | None = None,
        **kwargs: t.Any
    ):
        super().__init__(**kwargs)
        # Setup Widgets
        trait_vals = elt_traits.trait_values() if elt_traits else {}
        self.ui_elements : list[UIElement[t.Any, t.Any]] = [
            self.params_widget(
                template_map,
                trait_vals,
                RecipeKeys.INIT_PARAMS,
                RecipeLabels.INIT_PARAMS,
            ),
            self.params_widget(
                template_map,
                trait_vals,
                RecipeKeys.EXEC_PARAMS,
                RecipeLabels.EXEC_PARAMS,
            ),
        ]
        # Setup the Layout
        self.layout_ : mo.Html | None = (
            mo.vstack(self.ui_elements) if add_layout else None
        )

    def params_widget(
        self,
        template_map: dict[str, t.Any],
        trait_vals: dict[str, t.Any],
        params_key: str,
        params_label: str
    ):
        if template_map[params_key]:
            wd_itr : Iterable[UIElement[t.Any, t.Any] | None] = (
                self.property_widget(trait_vals, ekey, vmap)
                for ekey, vmap in template_map[params_key].items()
            ) 
            return mo.ui.array(
                [wx for wx in wd_itr if wx is not None],
                label=params_label
            )
        else:
            return mo.ui.array(
                [],
                label=params_label
            )

    def property_widget(
        self,
        traitv_dct: dict[str, t.Any],
        ekey: str, 
        vmap: dict[str, t.Any],
    ):
        if ekey in traitv_dct:
            vmap["default"] = traitv_dct[ekey]
        return render_property(
            vmap[RecipeKeys.TYPE],
            **vmap
        )


class StructSidePanel(PanelBase):
    def __init__(
        self,
        struct_comp: structure.StructBase,
        delay_build: bool = False, 
        **kwargs: t.Any
    ):
        super().__init__(**kwargs)
        self.struct_ : structure. StructBase = struct_comp
        if delay_build:
            self.ui_elements = []
            self.layout_ = None 
        else:
            self.build()

    def build(self):
        # Set Widgets
        wd_itr : Iterable[UIElement[t.Any, t.Any] | None] = (
            render_property(
                vmap.value_type,
                label=vmap.label  + " :",
                default=vmap.to_ui(self.struct_.get(ekey)),
            )
            for ekey, vmap in self.struct_.trait_ui().items()
        )
        self.ui_elements : list[UIElement[t.Any, t.Any]] = [
            wx for  wx in wd_itr if wx is not None
        ]
        self.layout_ : mo.Html | None = mo.vstack([
            mo.ui.array(
                self.ui_elements,
                label=self.struct_.name
            )
        ])

    @property
    @override
    def layout(self) -> mo.Html | None:
        if self.layout_ is not None:
            return self.layout_
        else:
            self.build()
            return self.layout_


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
            'state': {'disabled': False},
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

    def view_comps(self):
        return (self.tree, self.panel_dict)

    @staticmethod
    def panel_selector(
        axtree: awitree.Tree,
        axpanel_dict: dict[str, PanelBase]
    ) -> PanelBase | None:
        return (
            axpanel_dict[axtree.selected_nodes[0]["id"]]
            if (
                axtree.selected_nodes and
                len(axtree.selected_nodes) > 0 and
                (axtree.selected_nodes[0]["id"] in axpanel_dict)
            ) else None
        )


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

    def wf_step_render(
        self,
        wf_step: dict[str, t.Any]
    ) -> list[UIElement[t.Any, t.Any]]:
        step_key = wf_step[RecipeKeys.NAME]
        _log().debug("Initializing Panels for [%s]", step_key)
        ptemplate = self.mdr_setup.get_template_for(step_key)
        pload = recipe_step_payload(wf_step)
        return WorkflowStepPanel(
            ptemplate,
            False,
            pload.node_traits,
        ).ui_elements

    def db_recipe_node(
        self,
        db_key: str,
        db_desc: dict[str, t.Any]
    ) -> tuple[CBTreeNode, PanelBase]:
        db_node = CBTreeNode.init(
            name=db_desc[RecipeKeys.LABEL],
            key=db_key
        )
        wflow_desc = db_desc[RecipeKeys.WORKFLOW]
        _log().info(
            "Start Left-side panel construction for [%s]",
            str(db_key)
        )
        panel_layout = mo.vstack([
            mo.ui.array(
                self.wf_step_render(wf_step),
                label=f"Step {wf_idx + 1} : {wf_step[RecipeKeys.LABEL]}"
            )
            for wf_idx, wf_step in enumerate(wflow_desc)
        ])
        db_panel = PanelBase(layout=panel_layout)
        _log().info("Completed Left-side panel construction")
        return db_node, db_panel

    @override
    def build(self) -> TreeBase:
        self.tree : AnyWidget | None = self.build_tree()
        # self.tree.observe(self.tree_update, names="selected_nodes")
        self.layout_ : mo.Html | None = mo.hstack(
            [self.tree, mo.vstack([])],
            widths=[self.left_width, 1-self.left_width])
        return self

    @override
    @abc.abstractmethod
    def build_tree(self) -> AnyWidget:
        pass


class DataSourceRecipeView(RecipeTreeBase):
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
            wf_iter = itertools.chain(
                db_desc[RecipeKeys.DB_CONNECT][RecipeKeys.WORKFLOW],
                db_desc[RecipeKeys.POST_OPS][RecipeKeys.WORKFLOW],
            )
            db_node, db_panel = self.db_recipe_node(
                db_key,
                {
                    RecipeKeys.LABEL : db_desc[RecipeKeys.LABEL],
                    RecipeKeys.WORKFLOW : wf_iter
                }
            )
            self.panel_dict[db_key] = db_panel
            self.root_node.add_node(db_node)
        # initialize tree
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened" : True}
        tree : AnyWidget = awitree.Tree(data=tree_data)
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree


class Datat2ModelRecipeView(RecipeTreeBase):
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
        neuron_pfx : str = f"{location_name}-{neuron_name}"
        neuron_node = CBTreeNode.init(
            name=neuron_name,
            key=f"d2m_map-{neuron_pfx}"
        )
        for db_key, db_desc in neuron_desc[RecipeKeys.SRC_DATA].items():
            db_loc_key = f"{neuron_pfx}-{db_key}"
            db_node, db_panel = self.db_recipe_node(
                db_loc_key,
                db_desc
            )
            self.panel_dict[db_loc_key] = db_panel
            neuron_node.add_node(db_node)
        return neuron_node

    def locations(self):
        loc_root_node : CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.LOCATIONS,
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
            loc_root_node.add_node(loc_node)
        return loc_root_node

    def conncection_node(
        self,
        connect_name: str,
        connect_desc: dict[str, t.Any]
    ) -> CBTreeNode:
        conn_node = CBTreeNode.init(
            name=connect_name,
            key=f"d2m_map-{connect_name}"
        )
        for db_key, db_desc in connect_desc[RecipeKeys.SRC_DATA].items():
            db_conn_key = f"{connect_name}-{db_key}"
            db_node, db_panel = self.db_recipe_node(
                db_conn_key,
                db_desc
            )
            self.panel_dict[db_conn_key] = db_panel
            conn_node.add_node(db_node)
        return conn_node


    def connections(self):
        root_conn_node : CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.CONNECTIONS,
            key="d2m_map.connections.root"
        )
        for conn_name, conn_desc in self.d2m_map_desc[RecipeKeys.CONNECTIONS].items():
            root_conn_node.add_node(self.conncection_node(conn_name, conn_desc))
        return root_conn_node


    @override
    def build_tree(self) -> AnyWidget:
        self.root_node : CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.D2M,
            key="d2m_map.root"
        )
        self.root_node.add_node(self.locations())
        self.root_node.add_node(self.connections())
        # initialize tree
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened" : True}
        tree : AnyWidget = awitree.Tree(data=tree_data)
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree



class NetworkTreeView(TreeBase):
    def __init__(
        self,
        net_struct: structure.Network,
        left_width: float=0.4,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(left_width, **kwargs)
        self.net_struct : structure.Network = net_struct

    def region_node(
        self,
        net_region: structure.Region,
        region_pfx:str="r"
    ) -> CBTreeNode:
        region_key = f"{region_pfx}-{net_region.name}"
        region_node = CBTreeNode.from_struct(net_region, region_key)
        self.panel_dict[region_key] = StructSidePanel(net_region)
        for _, rx_neuron in net_region.neurons.items():
            neuron_key = f"{region_key}-{rx_neuron.name}"
            neuron_node = CBTreeNode.from_struct(rx_neuron, neuron_key)
            self.panel_dict[neuron_key] = StructSidePanel(rx_neuron)
            for _, nx_model in rx_neuron.neuron_models.items():
                model_key = f"{neuron_key}-{nx_model.name}"
                neuron_node.add_node(
                    CBTreeNode.from_struct(nx_model, model_key)
                )
                self.panel_dict[model_key] = StructSidePanel(nx_model, True)
            region_node.add_node(neuron_node)
        return region_node

    def connection_node(
        self,
        net_connect: structure.Connection,
        connect_pfx:str="c"
    ) -> CBTreeNode:
        conn_key = f"{connect_pfx}-{net_connect.name}"
        connect_node = CBTreeNode.from_struct(net_connect, conn_key)
        self.panel_dict[conn_key] = StructSidePanel(net_connect)
        for _, cx_model in net_connect.connect_models.items():
            cx_key = f"{conn_key}-{cx_model.name}"
            connect_node.add_node(CBTreeNode.from_struct(cx_model, cx_key))
            self.panel_dict[cx_key] = StructSidePanel(cx_model, True)
        return connect_node

    def ext_network_node(
            self,
            ext_net: structure.ExtNetwork,
            ext_prefix: str = ""
    ) -> CBTreeNode:
        ext_key = f"{ext_prefix}-{ext_net.name}"
        ext_net_node = CBTreeNode.from_struct(ext_net, ext_key)
        self.panel_dict[ext_key] = StructSidePanel(ext_net)
        location_node = CBTreeNode.init(
            StructTreeNames.REGIONS,
            ext_net.name + ".locations"
        )
        for _, net_region in ext_net.locations.items():
            location_node.add_node(self.region_node(net_region, ext_key))
        ext_net_node.add_node(location_node)
        connect_node = CBTreeNode.init(
            StructTreeNames.CONNECTIONS,
            ext_net.name + ".connections",
        )
        for _, net_connect in ext_net.connections.items():
            connect_node.add_node(self.connection_node(net_connect, ext_key))
        ext_net_node.add_node(connect_node)
        return ext_net_node

    def data_link_node(
            self,
            data_link: structure.DataLink,
            dl_prefix: str = ""
    ) -> CBTreeNode | None:
        if data_link.name and data_link.property_map:
            dl_key = f"{dl_prefix}-{data_link.name}"
            dl_node = CBTreeNode.from_struct(data_link, dl_key)
            self.panel_dict[dl_key] = StructSidePanel(data_link)
            return dl_node
        return None

    def data_file_node(
        self,
        data_file: structure.DataFile,
        df_prefix: str = ""
    ) -> CBTreeNode | None:
        df_key = f"{df_prefix}-{data_file.name}"
        df_node = CBTreeNode.from_struct(data_file, df_key)
        self.panel_dict[df_key] = StructSidePanel(data_file)
        return df_node

    def locations(self, root_key:str):
        #
        loc_key = f"{root_key}-loc"
        location_node = CBTreeNode.init(StructTreeNames.REGIONS, loc_key)
        #
        for _, net_region in self.net_struct.locations.items():
            location_node.add_node(self.region_node(net_region, loc_key))
        return location_node

    def connections(self, root_key: str):
        #
        conn_key = f"{root_key}-conn"
        connect_node = CBTreeNode.init(StructTreeNames.CONNECTIONS, conn_key)
        for _, net_connect in self.net_struct.connections.items():
            connect_node.add_node(self.connection_node(net_connect, conn_key))
        return connect_node

    def ext_networks(self, root_key: str):
        #
        ext_key = f"{root_key}-extnet"
        ext_net_node = CBTreeNode.init(
            StructTreeNames.EXTERNAL_NETWORKS,
            ext_key
        )
        for _, ext_net in self.net_struct.ext_networks.items():
            ext_net_node.add_node(self.ext_network_node(ext_net, ext_key))
        return ext_net_node

    def data_links(self, root_key: str):
        #
        data_link_key = f"{root_key}-dlinks"
        data_link_node = CBTreeNode.init(
            StructTreeNames.DATA_LINKS,
            data_link_key,
        )
        for data_cx in self.net_struct.data_connect:
            data_cx_node = self.data_link_node(data_cx, data_link_key)
            if data_cx_node:
                data_link_node.add_node(data_cx_node)
        return data_link_node

    def data_files(self, root_key: str):
        #
        data_file_key = f"{root_key}-dfiles"
        data_file_node = CBTreeNode.init(
            StructTreeNames.DATA_FILES,
            data_file_key,
        )
        for data_fx in self.net_struct.data_files:
            data_fx_node = self.data_file_node(data_fx, data_file_key)
            if data_fx_node:
                data_file_node.add_node(data_fx_node)
        return data_file_node

    def build_tree_nodes(self) -> CBTreeNode:
        root_key = self.net_struct.name
        #
        root_node = CBTreeNode.from_struct(self.net_struct, root_key)
        root_node.add_node(self.locations(root_key))
        root_node.add_node(self.connections(root_key))
        root_node.add_node(self.ext_networks(root_key))
        root_node.add_node(self.data_links(root_key))
        root_node.add_node(self.data_files(root_key))
        return root_node

    def build_tree(self) -> AnyWidget:
        self.root_node : CBTreeNode | None = self.build_tree_nodes()
        # initialize tree
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened" : True}
        tree : AnyWidget = awitree.Tree(data=tree_data)
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree


    @override
    def build(self) -> TreeBase:
        self.tree : AnyWidget | None = self.build_tree()
        # self.panel_dict : dict[str, PanelBase] = self.build_side_panels()
        # self.tree.observe(self.tree_update, names="selected_nodes")  # type:ignore
        self.layout_ : mo.Html | None = mo.hstack(
            [self.tree, mo.vstack([])],
            widths=self.widths
        )
        return self

    def values_dict(self):
        pass
