import abc
import itertools
import logging
import typing as t
import types
from typing_extensions import override
import ipywidgets as iwidgets
import ipytree as itree
import traitlets

from .. import base
from ..register import find_type
from ..model.setup import RecipeKeys, RecipeLabels, RecipeSetup
from ..model import structure as structure


def _log():
    return logging.getLogger(__name__)


@t.final
class CfgTreeNames:
    SRC_DATA = "Source Data"
    LOCATIONS = "Locations"
    CONNECTIONS = "Connections"


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
            self[ix, 0] = iwidgets.Label(str(kx) + " :")
            self[ix, 1] = self.entry_widget(kx, vx)

    def handle_change(self, change: dict[str, t.Any]):
        # print("L:", change)
        if change["name"] == "value" and change["old"] != change["new"]:
            _log().info("List CTR Change Value : " + str(change))
            node_key = change["owner"].node_key
            new_value = change["new"]
            self.value[node_key] = new_value
    
    def entry_widget(self, kx: int, vx: t.Any):
        tname = type(vx).__name__
        wdx = scalar_widget(tname, default=vx)
        wdx.add_traits(node_key=traitlets.Integer(kx))  # type: ignore
        wdx.observe(self.handle_change)  # type: ignore
        return wdx


class PropertyMapLayout(iwidgets.GridspecLayout):
    value : traitlets.Dict[str, t.Any] = traitlets.Dict(
        help="Dict values"
    ).tag(sync=True)

    def __init__(self, value: dict[str, t.Any], **kwargs: t.Any):
        # print("Value: ", len(value))
        if len(value) > 0:
            super().__init__(len(value), 2, value=value, **kwargs)
            for ix, (kx, vx) in enumerate(value.items()):
                self[ix, 0] = iwidgets.Label(str(kx) + " :")
                self[ix, 1] = self.entry_widget(kx, vx)
        else:
            super().__init__(1, 1, value=value, **kwargs)

    def handle_change(self, change: dict[str, t.Any]):
        # print("L:", change)
        if change["name"] == "value" and change["old"] != change["new"]:
            _log().info("List CTR Change Value : " + str(change))
            node_key = change["owner"].node_key
            new_value = change["new"]
            self.value[node_key] = new_value
 
    def init_widget(self, vx: t.Any) -> iwidgets.CoreWidget | None:
        tname = type(vx).__name__
        match tname:
            case "NoneType":
                return scalar_widget("str", default="None")
            case "list":
                return PropertyListLayout(vx)
            case _:
                return scalar_widget(tname, default=vx)

    def entry_widget(
        self,
        kx: str,
        vx: t.Any
    ) -> iwidgets.CoreWidget | None:
        # print("kx, vx: ", kx, vx)
        wdx = self.init_widget(vx)
        wdx.add_traits(node_key=traitlets.Unicode(kx))  # type: ignore
        wdx.observe(self.handle_change)  # type: ignore
        return wdx


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
class CBTreeNode(itree.Node):
    node_key : traitlets.Unicode[str, str | bytes] = traitlets.Unicode()


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

    def update(self, new_val: t.Any):
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
                iwidgets.Label(twid_def["label"] + " :"),
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
        struct_widgets = [iwidgets.Label(struct_comp.name)]
        struct_widgets += [
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
        self.panel_keys: set[str] = set([])
        self.panel_dict: dict[str, PanelBase] = {}
        self.left_width: str = left_width

    @abc.abstractmethod
    def init_tree(self) -> itree.Tree:
        return itree.Tree()

    @abc.abstractmethod
    def build(self) -> "TreeBase | None":
        return None

    def tree_update(self, change: dict[str, t.Any]):
        new_val = change["new"]
        if not len(new_val):
            return
        node_key = new_val[0].node_key
        _log().warning("Key : " + node_key + str(node_key in self.panel_dict))
        if node_key in self.panel_dict:
            side_panel = self.panel_dict[node_key]
            side_panel.update(new_val[0])
            self.layout.bottom_right = side_panel.layout

    @staticmethod
    def struct_tree_node(struct_obj: structure.StructBase) -> CBTreeNode:
        tnode_class = types.new_class(
            struct_obj.__class__.__name__ + "Node",
            bases=(CBTreeNode, struct_obj.trait_type()),
        )
        return tnode_class(
            node_key=struct_obj.name,
            **struct_obj.model_dump(exclude=struct_obj.exclude()),
        )

    @staticmethod
    def type_trait_tree_node(
        register_key: str,
        init_params: dict[str, t.Any]
    ) -> CBTreeNode | None:
        src_class: type[base.DbQuery] | type[base.OpXFormer] | None = find_type(
            register_key
        )
        if src_class:
            tnode_class = types.new_class(
                src_class.__name__ + "Node",
                bases=(CBTreeNode, src_class.trait_type()),
            )
            return tnode_class(**init_params)
        elif RecipeKeys.NODE_KEY in init_params and RecipeKeys.NAME in init_params:
            return CBTreeNode(**init_params)
        else:
            return None

    @staticmethod
    def wflow_step_tree_node(
        wf_step: dict[str, t.Any]
    ) -> CBTreeNode | None:
        step_key = wf_step[RecipeKeys.NAME]
        wf_dict = (
            {
                RecipeKeys.NAME: wf_step[RecipeKeys.LABEL],
                RecipeKeys.NODE_KEY: wf_step[RecipeKeys.NAME],
            }
            | wf_step[RecipeKeys.INIT_PARAMS]
            | wf_step[RecipeKeys.EXEC_PARAMS]
        )
        return RecipeTreeBase.type_trait_tree_node(step_key, wf_dict)


class RecipeTreeBase(TreeBase, metaclass=abc.ABCMeta):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str,
        **kwargs: t.Any,
    ):
        super().__init__(left_width, **kwargs)
        self.mdr_setup: RecipeSetup = mdr_setup

    def init_db_node(
        self, db_key: str, db_desc: dict[str, t.Any]
    ) -> CBTreeNode:
        db_node = CBTreeNode(name=db_desc[RecipeKeys.LABEL], node_key=db_key)
        for wf_step in db_desc[RecipeKeys.WORKFLOW]:
            db_node.add_node(RecipeTreeBase.wflow_step_tree_node(wf_step))
            self.panel_keys.add(wf_step[RecipeKeys.NAME])
        return db_node

    def init_side_panels(self) -> dict[str, PanelBase]:
        _log().info(
            "Start Left-side panel construction for [%s]", str(self.panel_dict.keys())
        )
        for pkey in self.panel_keys:
            _log().debug("Initializing Panels for [%s]", pkey)
            ptemplate = self.mdr_setup.get_template_for(pkey)
            self.panel_dict[pkey] = RecipeSidePanel(ptemplate)
        _log().info("Completed Left-side panel construction")
        return self.panel_dict

    @override
    def build(self) -> TreeBase:
        self.tree : itree.Tree = self.init_tree()
        self.panel_dict : dict[str, PanelBase]= self.init_side_panels()
        self.tree.observe(self.tree_update, names="selected_nodes")  # type: ignore
        self.layout : iwidgets.TwoByTwoLayout = iwidgets.TwoByTwoLayout(
            top_left=self.tree,
            bottom_right=None
        )
        return self

    @override
    @abc.abstractmethod
    def init_tree(self) -> itree.Tree:
        return itree.Tree()


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
    def init_tree(self) -> itree.Tree:
        root_node = CBTreeNode(name=CfgTreeNames.SRC_DATA, node_key="source_data")
        for db_key, db_desc in self.src_data_desc.items():
            db_node = self.init_db_node(
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
        self.tree : itree.Tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type:ignore
        return self.tree


class D2MLocationsTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.d2m_map_desc : dict[str, t.Any] = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]

    def init_neuron_node(
        self, neuron_name: str, neuron_desc: dict[str, t.Any]
    ) -> CBTreeNode:
        neuron_node = CBTreeNode(name=neuron_name, node_key="d2m_map.neuron")
        for db_key, db_desc in neuron_desc[RecipeKeys.SRC_DATA].items():
            neuron_node.add_node(self.init_db_node(db_key, db_desc))
        return neuron_node

    @override
    def init_tree(self) -> itree.Tree:
        root_node = CBTreeNode(name=RecipeKeys.LOCATIONS, node_key="root")
        for loc_name, loc_desc in self.d2m_map_desc[RecipeKeys.LOCATIONS].items():
            loc_node = CBTreeNode(name=loc_name, node_key="d2m_map.location")
            for neuron_name, neuron_desc in loc_desc.items():
                neuron_node = self.init_neuron_node(neuron_name, neuron_desc)
                loc_node.add_node(neuron_node)
            root_node.add_node(loc_node)
        self.tree : itree.Tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type: ignore
        return self.tree


class D2MConnectionsTreeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.d2m_map_desc : dict[str, t.Any] = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]

    def init_conncection_node(
        self, connect_name: str, connect_desc: dict[str, t.Any]
    ) -> CBTreeNode:
        conn_node = CBTreeNode(name=connect_name, node_key="d2m_map.connection")
        for db_key, db_desc in connect_desc[RecipeKeys.SRC_DATA].items():
            db_node = CBTreeNode(name=db_desc[RecipeKeys.LABEL], node_key=db_key)
            for wf_step in db_desc[RecipeKeys.WORKFLOW]:
                db_node.add_node(RecipeTreeBase.wflow_step_tree_node(wf_step))
            self.panel_keys.union(
                set(
                    wf_step[RecipeKeys.NAME] for wf_step in db_desc[RecipeKeys.WORKFLOW]
                )
            )
            conn_node.add_node(self.init_db_node(db_key, db_desc))
        return conn_node

    @override
    def init_tree(self) -> itree.Tree:
        root_node = CBTreeNode(name=CfgTreeNames.CONNECTIONS, node_key="tree")
        for conn_name, conn_desc in self.d2m_map_desc[RecipeKeys.CONNECTIONS].items():
            root_node.add_node(self.init_conncection_node(conn_name, conn_desc))
        self.tree : itree.Tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type: ignore
        return self.tree


class NetworkTreeView(TreeBase):
    def __init__(
        self,
        net_struct: structure.Network,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(left_width, **kwargs)
        self.net_struct : structure.Network = net_struct

    def init_region_side_panels(self, net_region: structure.Region) -> None:
        self.panel_dict[net_region.name] = StructSidePanel(net_region)
        for _, rx_neuron in net_region.neurons.items():
            self.panel_dict[rx_neuron.name] = StructSidePanel(rx_neuron)
            for _, nx_model in rx_neuron.neuron_models.items():
                self.panel_dict[nx_model.name] = StructSidePanel(nx_model)

    def init_conn_side_panels(self, net_connect: structure.Connection) -> None:
        self.panel_dict[net_connect.name] = StructSidePanel(net_connect)
        for _, cx_model in net_connect.connect_models.items():
            self.panel_dict[cx_model.name] = StructSidePanel(cx_model)

    # def init_ext_net_side_panels(self, ext_net: structure.ExtNetwork) -> None:
    #     self.panel_dict[ext_net.name] = StructSidePanel(ext_net)
    #     for _, net_region in ext_net.locations.items():
    #         self.init_region_side_panels(net_region)
    #     for _, net_connect in ext_net.connections.items():
    #         self.init_conn_side_panels(net_connect)

    def init_graph_side_panels(self, gnet: structure.ExtNetwork | structure.Network) -> None:
        self.panel_dict[gnet.name] = StructSidePanel(gnet)
        #
        for _, net_region in gnet.locations.items():
            self.init_region_side_panels(net_region)
        #
        for _, net_connect in gnet.connections.items():
            self.init_conn_side_panels(net_connect)

    def init_ext_network_panels(self) -> None:
        for _, ext_net in self.net_struct.ext_networks.items():
            self.init_graph_side_panels(ext_net)

    def init_data_link_panels(self) -> None:
        for data_lx in self.net_struct.data_connect:
            if data_lx.name and data_lx.property_map:
                self.panel_dict[data_lx.name] = StructSidePanel(data_lx)

    def init_data_file_panels(self) -> None:
        for data_fx in self.net_struct.data_files:
            self.panel_dict[data_fx.name] = StructSidePanel(data_fx)


    def init_side_panels(self) -> dict[str, PanelBase]:
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
        self.init_graph_side_panels(self.net_struct) 
        self.init_ext_network_panels()
        self.init_data_link_panels()
        self.init_data_file_panels()
        _log().info("Completed Left-side panel construction")
        return self.panel_dict

    def region_node(self, net_region: structure.Region) -> CBTreeNode:
        region_node = TreeBase.struct_tree_node(net_region)
        for _, rx_neuron in net_region.neurons.items():
            neuron_node = TreeBase.struct_tree_node(rx_neuron)
            for _, nx_model in rx_neuron.neuron_models.items():
                neuron_node.add_node(TreeBase.struct_tree_node(nx_model))
            region_node.add_node(neuron_node)
        return region_node

    def connection_node(self, net_connect: structure.Connection) -> CBTreeNode:
        connect_node = TreeBase.struct_tree_node(net_connect)
        for _, cx_model in net_connect.connect_models.items():
            connect_node.add_node(TreeBase.struct_tree_node(cx_model))
        return connect_node

    def ext_network_node(self, ext_net: structure.ExtNetwork) -> CBTreeNode:
        ext_net_node = TreeBase.struct_tree_node(ext_net)
        location_node = CBTreeNode(node_key=ext_net.name + ".locations", name="Regions")
        for _, net_region in ext_net.locations.items():
            location_node.add_node(self.region_node(net_region))
        ext_net_node.add_node(location_node)
        connect_node = CBTreeNode(
            node_key=ext_net.name + ".connections", name="Connections"
        )
        for _, net_connect in ext_net.connections.items():
            connect_node.add_node(self.connection_node(net_connect))
        ext_net_node.add_node(connect_node)
        return ext_net_node

    def data_link_node(self, data_link: structure.DataLink) -> CBTreeNode | None:
        if data_link.name and data_link.property_map:
            return TreeBase.struct_tree_node(data_link)
        return None

    def data_file_node(self, data_file: structure.DataFile) -> CBTreeNode | None:
        return TreeBase.struct_tree_node(data_file)

    def init_tree_nodes(self) -> CBTreeNode:
        root_node = TreeBase.struct_tree_node(self.net_struct)
        location_node = CBTreeNode(node_key="net.locations", name="Regions")
        #
        for _, net_region in self.net_struct.locations.items():
            location_node.add_node(self.region_node(net_region))
        root_node.add_node(location_node)
        #
        connect_node = CBTreeNode(node_key="net.connections", name="Connections")
        for _, net_connect in self.net_struct.connections.items():
            connect_node.add_node(self.connection_node(net_connect))
        root_node.add_node(connect_node)
        #
        ext_net_node = CBTreeNode(node_key="net.ext_networks", name="External Networks")
        for _, ext_net in self.net_struct.ext_networks.items():
            ext_net_node.add_node(self.ext_network_node(ext_net))
        root_node.add_node(ext_net_node)
        #
        data_link_node = CBTreeNode(node_key="net.data_links", name="Data Links")
        for data_cx in self.net_struct.data_connect:
            data_link_node.add_node(self.data_link_node(data_cx))
        root_node.add_node(data_link_node)
        #
        data_file_node = CBTreeNode(node_key="net.data_files", name="Data Files")
        for data_fx in self.net_struct.data_files:
            data_link_node.add_node(self.data_file_node(data_fx))
        root_node.add_node(data_file_node)
        return root_node

    @override
    def init_tree(self) -> itree.Tree:
        root_node = self.init_tree_nodes()
        self.tree : itree.Tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type: ignore
        return self.tree

    @override
    def build(self) -> TreeBase:
        self.tree = self.init_tree()
        self.panel_dict : dict[str, PanelBase] = self.init_side_panels()
        self.tree.observe(self.tree_update, names="selected_nodes")  # type:ignore
        self.layout : iwidgets.TwoByTwoLayout = iwidgets.TwoByTwoLayout(
            top_left=self.tree, bottom_right=None
        )
        return self

    def values_dict(self):
        pass
