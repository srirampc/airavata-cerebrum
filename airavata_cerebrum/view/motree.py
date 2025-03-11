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


class D2MConnectionsTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: float=0.4,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.d2m_map_desc : dict[str, t.Any] = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]

    def conncection_node(
        self,
        connect_name: str,
        connect_desc: dict[str, t.Any]
    ) -> CBTreeNode:
        conn_node = CBTreeNode.init(
            name=connect_name,
            key=f"d2m_map.connection-{connect_name}"
        )
        for db_key, db_desc in connect_desc[RecipeKeys.SRC_DATA].items():
            db_node = self.db_recipe_node(
                f"{connect_name}-{db_key}",
                db_desc
            )
            conn_node.add_node(db_node)
        return conn_node

    @override
    def build_tree(self) -> AnyWidget:
        self.root_node : CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.CONNECTIONS,
            key="d2m_map.connections.root"
        )
        for conn_name, conn_desc in self.d2m_map_desc[RecipeKeys.CONNECTIONS].items():
            self.root_node.add_node(self.conncection_node(conn_name, conn_desc))
        # initialize tree
        tree : AnyWidget = awitree.Tree(
            data=self.root_node.awi_dict()
        )
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

    def region_side_panels(self, net_region: structure.Region) -> None:
        self.panel_dict[net_region.name] = StructSidePanel(net_region)
        for _, rx_neuron in net_region.neurons.items():
            self.panel_dict[rx_neuron.name] = StructSidePanel(rx_neuron)
            for _, nx_model in rx_neuron.neuron_models.items():
                self.panel_dict[nx_model.name] = StructSidePanel(nx_model)

    def conn_side_panels(self, net_connect: structure.Connection) -> None:
        self.panel_dict[net_connect.name] = StructSidePanel(net_connect)
        for _, cx_model in net_connect.connect_models.items():
            self.panel_dict[cx_model.name] = StructSidePanel(cx_model)

    # def init_ext_net_side_panels(self, ext_net: structure.ExtNetwork) -> None:
    #     self.panel_dict[ext_net.name] = StructSidePanel(ext_net)
    #     for _, net_region in ext_net.locations.items():
    #         self.init_region_side_panels(net_region)
    #     for _, net_connect in ext_net.connections.items():
    #         self.init_conn_side_panels(net_connect)

    def graph_side_panels(self, gnet: structure.ExtNetwork | structure.Network) -> None:
        self.panel_dict[gnet.name] = StructSidePanel(gnet)
        #
        for _, net_region in gnet.locations.items():
            self.region_side_panels(net_region)
        #
        for _, net_connect in gnet.connections.items():
            self.conn_side_panels(net_connect)

    def ext_network_panels(self) -> None:
        for _, ext_net in self.net_struct.ext_networks.items():
            self.graph_side_panels(ext_net)

    def data_link_panels(self) -> None:
        for data_lx in self.net_struct.data_connect:
            if data_lx.name and data_lx.property_map:
                self.panel_dict[data_lx.name] = StructSidePanel(data_lx)

    def data_file_panels(self) -> None:
        for data_fx in self.net_struct.data_files:
            self.panel_dict[data_fx.name] = StructSidePanel(data_fx)


    def build_side_panels(self) -> dict[str, PanelBase]:
        _log().info(
            "Start Right-side panel construction for [%s]",
            str(self.net_struct.name)
        )
        # self.panel_dict[self.net_struct.name] = StructSidePanel(self.net_struct)
        # #
        # for _, net_region in self.net_struct.locations.items():
        #     self.init_region_side_panels(net_region)
        # #
        # for _, net_connect in self.net_struct.connections.items():
        #     self.init_conn_side_panels(net_connect)
        self.graph_side_panels(self.net_struct) 
        self.ext_network_panels()
        self.data_link_panels()
        self.data_file_panels()
        _log().info("Completed Left-side panel construction")
        return self.panel_dict

    def region_node(self, net_region: structure.Region) -> CBTreeNode:
        region_node = CBTreeNode.from_struct(net_region)
        for _, rx_neuron in net_region.neurons.items():
            neuron_node = CBTreeNode.from_struct(rx_neuron)
            for _, nx_model in rx_neuron.neuron_models.items():
                neuron_node.add_node(CBTreeNode.from_struct(nx_model))
            region_node.add_node(neuron_node)
        return region_node

    def connection_node(self, net_connect: structure.Connection) -> CBTreeNode:
        connect_node = CBTreeNode.from_struct(net_connect)
        for _, cx_model in net_connect.connect_models.items():
            connect_node.add_node(CBTreeNode.from_struct(cx_model))
        return connect_node

    def ext_network_node(self, ext_net: structure.ExtNetwork) -> CBTreeNode:
        ext_net_node = CBTreeNode.from_struct(ext_net)
        location_node = CBTreeNode.init(
            StructTreeNames.REGIONS,
            ext_net.name + ".locations"
        )
        for _, net_region in ext_net.locations.items():
            location_node.add_node(self.region_node(net_region))
        ext_net_node.add_node(location_node)
        connect_node = CBTreeNode.init(
            StructTreeNames.CONNECTIONS,
            ext_net.name + ".connections",
        )
        for _, net_connect in ext_net.connections.items():
            connect_node.add_node(self.connection_node(net_connect))
        ext_net_node.add_node(connect_node)
        return ext_net_node

    def data_link_node(self, data_link: structure.DataLink) -> CBTreeNode | None:
        if data_link.name and data_link.property_map:
            return CBTreeNode.from_struct(data_link)
        return None

    def data_file_node(self, data_file: structure.DataFile) -> CBTreeNode | None:
        return CBTreeNode.from_struct(data_file)

    def build_tree_nodes(self) -> CBTreeNode:
        root_node = CBTreeNode.from_struct(self.net_struct)
        location_node = CBTreeNode.init(
            StructTreeNames.REGIONS,
            "net.locations"
        )
        #
        for _, net_region in self.net_struct.locations.items():
            location_node.add_node(self.region_node(net_region))
        root_node.add_node(location_node)
        #
        connect_node = CBTreeNode.init(
            StructTreeNames.CONNECTIONS,
            "net.connections",
        )
        for _, net_connect in self.net_struct.connections.items():
            connect_node.add_node(self.connection_node(net_connect))
        root_node.add_node(connect_node)
        #
        ext_net_node = CBTreeNode.init(
            StructTreeNames.EXTERNAL_NETWORKS,
             "net.ext_networks",
        )
        for _, ext_net in self.net_struct.ext_networks.items():
            ext_net_node.add_node(self.ext_network_node(ext_net))
        root_node.add_node(ext_net_node)
        #
        data_link_node = CBTreeNode.init(
            StructTreeNames.DATA_LINKS,
            "net.data_links",
        )
        for data_cx in self.net_struct.data_connect:
            data_cx_node = self.data_link_node(data_cx)
            if data_cx_node:
                data_link_node.add_node(data_cx_node)
        root_node.add_node(data_link_node)
        #
        data_file_node = CBTreeNode.init(
            StructTreeNames.DATA_FILES,
            "net.data_files",
        )
        for data_fx in self.net_struct.data_files:
            data_fx_node = self.data_file_node(data_fx)
            if data_fx_node:
                data_link_node.add_node(data_fx_node)
        root_node.add_node(data_file_node)
        return root_node

    def build_tree(self) -> AnyWidget:
        self.root_node : CBTreeNode | None = self.build_tree_nodes()
        # initialize tree
        tree : AnyWidget = awitree.Tree(
            data=self.root_node.awi_dict()
        )
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree


    @override
    def build(self) -> TreeBase:
        self.tree : AnyWidget | None = self.build_tree()
        self.panel_dict : dict[str, PanelBase] = self.build_side_panels()
        # self.tree.observe(self.tree_update, names="selected_nodes")  # type:ignore
        self.layout : mo.Html | None = mo.hstack(
            [self.tree, mo.vstack([])],
            widths=self.widths
        )
        return self

    def values_dict(self):
        pass
