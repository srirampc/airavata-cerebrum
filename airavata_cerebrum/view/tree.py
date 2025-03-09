import abc
import itertools
import logging
import typing as t
from typing_extensions import override
import ipywidgets as iwidgets
import ipytree as itree
import traitlets

from ..model.setup import RecipeKeys, RecipeLabels, RecipeSetup
from ..model import structure as structure
from . import RcpTreeNames, StructTreeNames, PayLoad, recipe_step_payload, struct_payload


def _log():
    return logging.getLogger(__name__)


def scalar_widget(
    widget_key: str,
    **kwargs: t.Any
) -> iwidgets.CoreWidget | None:
    match widget_key:
        case "int" | "int32" | "int64":
            return iwidgets.IntText(value=kwargs["default"], disabled=False)
        case "float" | "float32" | "float64":
            return iwidgets.FloatText(value=kwargs["default"], disabled=False)
        case "text":
            return iwidgets.Text(value=kwargs["default"], disabled=False)
        case "textarea" | "str":
            return iwidgets.Textarea(value=kwargs["default"], disabled=False)
        case "option":
            return iwidgets.Dropdown(
                options=kwargs["options"],
                disabled=False,
            )
        case "check" | "bool":
            return iwidgets.Checkbox(
                value=bool(kwargs["default"]),
                disabled=False,
                indent=True,
            )
        case "tags":
            return iwidgets.TagsInput(
                value=kwargs["default"],
                allowed_tags=kwargs["allowed"],
                allowed_duplicates=False,
            )
        case _:
            return None


class PropertyListLayout(iwidgets.GridspecLayout):
    value : traitlets.List[t.Any] = traitlets.List(
        help="List values"
    ).tag(sync=True)

    def __init__(self, value: list[t.Any], **kwargs: t.Any):
        super().__init__(len(value), 2, value=value, **kwargs)
        for ix, (kx, vx) in enumerate(enumerate(value)):
            wdx = scalar_widget(type(vx).__name__, default=vx)
            wdx.add_traits(index=traitlets.Integer(kx))
            wdx.observe(self.handle_change)
            self[ix, 0] = iwidgets.Label(str(kx) + " :")
            self[ix, 1] = wdx

    def handle_change(self, change: dict[str, t.Any]):
        # print("L:", change)
        if change["name"] == "value" and change["old"] != change["new"]:
            _log().info("List CTR Change Value : " + str(change))
            entry_index = change["owner"].index
            new_value = change["new"]
            self.value[entry_index] = new_value


class PropertyMapLayout(iwidgets.GridspecLayout):
    value : traitlets.Dict = traitlets.Dict(
        help="Dict values"
    ).tag(sync=True)

    def __init__(self, value: dict[str, t.Any], **kwargs: t.Any):
        # print("Value: ", len(value))
        if len(value) > 0:
            super().__init__(len(value), 2, value=value, **kwargs)
            for ix, (kx, vx) in enumerate(value.items()):
                wdx = self.init_widget(vx, type(vx).__name__)
                wdx.add_traits(key=traitlets.Unicode(kx))  # type: ignore
                wdx.observe(self.handle_change)  # type: ignore
                self[ix, 0] = iwidgets.Label(str(kx) + " :")
                self[ix, 1] = wdx 
        else:
            super().__init__(1, 1, value=value, **kwargs)

    def handle_change(self, change: dict[str, t.Any]):
        # print("L:", change)
        if change["name"] == "value" and change["old"] != change["new"]:
            _log().info("List CTR Change Value : " + str(change))
            entry_key = change["owner"].key
            new_value = change["new"]
            self.value[entry_key] = new_value
 
    def init_widget(self, vx: t.Any, tname: str) -> iwidgets.CoreWidget | None:
        match tname:
            case "NoneType":
                return scalar_widget("str", default="None")
            case "list":
                return PropertyListLayout(vx)
            case _:
                return scalar_widget(tname, default=vx)


def render_property(
    widget_key: str,
    **kwargs: t.Any
) -> iwidgets.CoreWidget | None:
    match widget_key:
        case "dict":
            return PropertyMapLayout(value=kwargs["default"], **kwargs)
        case "list":
            return PropertyListLayout(value=kwargs["default"], **kwargs)
        case _:
            return scalar_widget(widget_key, **kwargs)


#
# Base class for tree node
#  rnode = CBTreeNode(name='a', node=TNode(key='k', traits=traitlets.Dict({'a':4})))
#  rnode.add_node(CBTreeNode(name='b', node=TNode(key='j', traits=traitlets.Dict({'b':4}))))
#  tree = itree.Tree()
#  tree.add_node(rnode)
#  tree
class CBTreeNode(itree.Node):

    def __init__(
        self,
        name: str,
        payload: PayLoad | None,
        nodes: tuple[t.Any] | None= None,
        **kwargs : t.Any
    ):
        super().__init__(name, nodes if nodes else [], **kwargs)
        self.payload : PayLoad | None = payload

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

# Base class for side panel
class PanelBase:
    def __init__(self, **kwargs: t.Any):
        self.layout : iwidgets.Box | None = None
        self.widget_map : dict[str, iwidgets.CoreWidget | None] = {}
        self.links : list[iwidgets.link] = []

    def clear_links(self):
        for lx in self.links:
            lx.unlink()
        self.links.clear()

    def update(self, new_val: traitlets.HasTraits | None):
        self.clear_links()
        if not new_val:
            return
        for slot, widx in self.widget_map.items():
            self.links.append(iwidgets.link((new_val, slot), (widx, "value")))


class RecipeSidePanel(PanelBase):
    def __init__(self, template_map: dict[str, t.Any], **kwargs: t.Any):
        super().__init__(**kwargs)
        for ekey, vmap in template_map[RecipeKeys.INIT_PARAMS].items():
            self.widget_map[ekey] = render_property(vmap[RecipeKeys.TYPE], **vmap)
        for ekey, vmap in template_map[RecipeKeys.EXEC_PARAMS].items():
            self.widget_map[ekey] = render_property(vmap[RecipeKeys.TYPE], **vmap)

        # Set Widgets
        ip_widgets: list[iwidgets.Label | iwidgets.HBox] = []
        if (
            RecipeKeys.INIT_PARAMS in template_map
            and template_map[RecipeKeys.INIT_PARAMS]
        ):
            ip_widgets = [iwidgets.Label(RecipeLabels.INIT_PARAMS)] + [
                self.wrap(kx, wx)
                for kx, wx in template_map[RecipeKeys.INIT_PARAMS].items()
            ]
        else:
            ip_widgets = [iwidgets.Label(RecipeLabels.INIT_PARAMS + RecipeLabels.NA)]
        ep_widgets: list[iwidgets.Label | iwidgets.HBox] = []
        if (
            RecipeKeys.EXEC_PARAMS in template_map
            and template_map[RecipeKeys.EXEC_PARAMS]
        ):
            ep_widgets = [iwidgets.Label(RecipeLabels.EXEC_PARAMS)] + [
                self.wrap(kx, wx)
                for kx, wx in template_map[RecipeKeys.EXEC_PARAMS].items()
            ]
        else:
            ep_widgets = [iwidgets.Label(RecipeLabels.EXEC_PARAMS + RecipeLabels.NA)]
        self.layout : iwidgets.Box | None = iwidgets.VBox(
            ip_widgets + ep_widgets
        )

    def wrap(self, widget_key: str, twid_def: dict[str, str]):
        # print("W Key", widget_key)
        return iwidgets.HBox(
            [
                iwidgets.Label(twid_def[RecipeKeys.LABEL] + " :"),
                self.widget_map[widget_key],
            ]
        )


class StructSidePanel(PanelBase):
    def __init__(self, struct_comp: structure.StructBase, **kwargs: t.Any):
        super().__init__(**kwargs)
        str_dict = struct_comp.model_dump()
        for ekey, vmap in struct_comp.trait_ui().items():
            self.widget_map[ekey] = render_property(
                vmap.value_type,
                default=vmap.to_ui(str_dict[ekey]),
            )

        # Set Widgets
        struct_widgets = [iwidgets.Label(struct_comp.name)] + [
            self.wrap(kx, wx) for kx, wx in struct_comp.trait_ui().items()
        ]
        self.layout : iwidgets.Box | None = iwidgets.VBox(struct_widgets)

    def wrap(self, widget_key: str, twid_dict : structure.TraitDef):
        return iwidgets.HBox(
            [
                iwidgets.Label(twid_dict.label + " :"),
                self.widget_map[widget_key],
            ]
        )


class TreeBase(abc.ABC):
    def __init__(
        self,
        left_width: str,
        **kwargs: t.Any,
    ):
        self.tree: itree.Tree = itree.Tree()
        self.layout: iwidgets.TwoByTwoLayout = iwidgets.TwoByTwoLayout()
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
            self.layout.bottom_right = side_panel.layout


class RecipeTreeBase(TreeBase, metaclass=abc.ABCMeta):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str,
        **kwargs: t.Any,
    ):
        super().__init__(left_width, **kwargs)
        self.panel_keys: set[str] = set([])
        self.mdr_setup: RecipeSetup = mdr_setup

    def recipe_step_node(
        self, db_key: str, db_desc: dict[str, t.Any]
    ) -> tuple[CBTreeNode, set[str]]:
        panel_keys = set([])
        db_node = CBTreeNode.init(db_desc[RecipeKeys.LABEL], db_key)
        for wf_step in db_desc[RecipeKeys.WORKFLOW]:
            rcp_node = CBTreeNode.from_recipe_step(wf_step)
            # _log().warning("Add %s %s %s",
            #                str(wf_step[RecipeKeys.NAME]),
            #                str(rcp_node.payload.node_key),
            #                str(rcp_node.payload.node_traits.trait_names()))
            db_node.add_node(rcp_node)
            panel_keys.add(wf_step[RecipeKeys.NAME])
        return db_node, panel_keys

    @override
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

    @abc.abstractmethod
    def build_tree(self) -> tuple[itree.Tree, set[str]]:
        pass

    @override
    def build(self) -> "TreeBase | None":
        tree, self.panel_keys  = self.build_tree()
        self.tree : itree.Tree = tree
        self.panel_dict : dict[str, PanelBase] = self.build_side_panels()
        self.tree.observe(self.tree_update, names="selected_nodes")
        self.layout : iwidgets.TwoByTwoLayout = iwidgets.TwoByTwoLayout(
            top_left=self.tree, bottom_right=None
        )
        return self



class SourceDataTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.src_data_desc : dict[str, t.Any] = mdr_setup.recipe_sections[RecipeKeys.SRC_DATA]

    @override
    def build_tree(self) -> tuple[itree.Tree, set[str]]:
        panel_keys:  set[str] = set([])
        root_node = CBTreeNode.init(RcpTreeNames.SRC_DATA, "source_data")
        for db_key, db_desc in self.src_data_desc.items():
            db_node, node_panel_keys = self.recipe_step_node(
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
            panel_keys |= node_panel_keys
        tree : itree.Tree = itree.Tree(multiple_selection=False)
        tree.add_node(root_node)
        tree.layout.width = self.left_width  # type:ignore
        return tree, panel_keys


class D2MLocationsTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.d2m_map_desc : dict[str, t.Any] = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]

    def neuron_node(
        self, neuron_name: str, neuron_desc: dict[str, t.Any]
    ) -> tuple[CBTreeNode, set[str]]:
        panel_keys : set[str] = set([])
        neuron_node = CBTreeNode.init(neuron_name, RcpTreeNames.D2M_NEURON)
        for db_key, db_desc in neuron_desc[RecipeKeys.SRC_DATA].items():
            db_node, node_panel_keys = self.recipe_step_node(db_key, db_desc)
            neuron_node.add_node(db_node)
            panel_keys |= node_panel_keys
        return neuron_node, panel_keys

    @override
    def build_tree(self) -> tuple[itree.Tree, set[str]]:
        panel_keys : set[str] = set([])
        root_node = CBTreeNode.init(RecipeKeys.LOCATIONS, "root")
        for loc_name, loc_desc in self.d2m_map_desc[RecipeKeys.LOCATIONS].items():
            loc_node = CBTreeNode.init(loc_name, RcpTreeNames.D2M_LOCATION)
            for neuron_name, neuron_desc in loc_desc.items():
                neuron_node, node_panel_keys = self.neuron_node(
                    neuron_name,
                    neuron_desc
                )
                loc_node.add_node(neuron_node)
                panel_keys |= node_panel_keys
            root_node.add_node(loc_node)
        tree : itree.Tree = itree.Tree(multiple_selection=False)
        tree.add_node(root_node)
        tree.layout.width = self.left_width
        return tree, panel_keys


class D2MConnectionsTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.d2m_map_desc : dict[str, t.Any] = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]

    def conncection_node(
        self, connect_name: str, connect_desc: dict[str, t.Any]
    ) -> tuple[CBTreeNode, set[str]]:
        panel_keys : set[str] = set([])
        conn_node = CBTreeNode.init(connect_name, RcpTreeNames.D2M_CONNECION)
        for db_key, db_desc in connect_desc[RecipeKeys.SRC_DATA].items():
            db_node = CBTreeNode.init(db_desc[RecipeKeys.LABEL], db_key)
            for wf_step in db_desc[RecipeKeys.WORKFLOW]:
                db_node.add_node(CBTreeNode.from_recipe_step(wf_step))
            panel_keys |= set(
                wf_step[RecipeKeys.NAME]
                for wf_step in db_desc[RecipeKeys.WORKFLOW]
            )
            conn_node.add_node(db_node)
        return conn_node, panel_keys

    @override
    def build_tree(self) -> tuple[itree.Tree, set[str]]:
        panel_keys : set[str] = set([])
        root_node = CBTreeNode(
            name=RcpTreeNames.CONNECTIONS,
            payload=PayLoad("tree")
        )
        for name, desc in self.d2m_map_desc[RecipeKeys.CONNECTIONS].items():
            conn_node, node_panel_keys = self.conncection_node(name, desc)
            root_node.add_node(conn_node)
            panel_keys |= node_panel_keys
        tree : itree.Tree = itree.Tree(multiple_selection=False)
        tree.add_node(root_node)
        tree.layout.width = self.left_width  # type: ignore
        return tree, panel_keys


class NetworkTreeView(TreeBase):
    def __init__(
        self,
        net_struct: structure.Network,
        left_width: str="40%",
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
            "Start Right-side panel construction for [%s]", str(self.net_struct.name)
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
            data_link_node.add_node(self.data_link_node(data_cx))
        root_node.add_node(data_link_node)
        #
        data_file_node = CBTreeNode.init(
            StructTreeNames.DATA_FILES,
            "net.data_files",
        )
        for data_fx in self.net_struct.data_files:
            data_link_node.add_node(self.data_file_node(data_fx))
        root_node.add_node(data_file_node)
        return root_node

    def build_tree(self) -> itree.Tree:
        root_node = self.build_tree_nodes()
        self.tree : itree.Tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type: ignore
        return self.tree

    @override
    def build(self) -> TreeBase:
        self.tree = self.build_tree()
        self.panel_dict : dict[str, PanelBase] = self.build_side_panels()
        self.tree.observe(self.tree_update, names="selected_nodes")  # type:ignore
        self.layout : iwidgets.TwoByTwoLayout = iwidgets.TwoByTwoLayout(
            top_left=self.tree, bottom_right=None
        )
        return self

    def values_dict(self):
        pass
