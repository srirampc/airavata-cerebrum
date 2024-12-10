import abc
import itertools
import logging
import typing
import types
import ipywidgets as iwidgets
import ipytree as itree
import traitlets
from traitlets.traitlets import Integer

from .. import base
from ..register import find_type
from ..model.setup import RecipeKeys, RecipeLabels, RecipeSetup
from ..model import structure as structure


def _log():
    return logging.getLogger(__name__)


class CfgTreeNames:
    SRC_DATA = "Source Data"
    LOCATIONS = "Locations"
    CONNECTIONS = "Connections"


def scalar_widget(widget_key, **kwargs) -> None | iwidgets.CoreWidget:
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


class ContainerLayout(iwidgets.GridspecLayout):
    def __init__(self, n_rows, n_columns, value, **kwargs):
        super().__init__( n_rows, n_columns, value=value, **kwargs)
        if n_columns == 2:
            for ix, (kx, vx) in enumerate(self.value_items(value)):
                self[ix, 0] = iwidgets.Label(str(kx) + " :")
                self[ix, 1] = self.entry_widget(kx, vx)
        else:
            self[0, 0] = iwidgets.Label(" --N/A-- ")
        # self.observe(self.change_value)  # type: ignore

    @abc.abstractmethod
    def value_items(self, value) -> typing.Iterable:
        pass

    @abc.abstractmethod
    def entry_widget(self, kx, vx) -> iwidgets.Widget | None:
        pass

    @abc.abstractmethod
    def update_value(self, node_key, new_value) -> None:
        pass

    def change_value(self, change):
        # print("L:", change)
        if change["name"] == "value" and change["old"] != change["new"]:
            node_key = change["owner"].node_key
            new_value = change["new"]
            self.update_value(node_key, new_value)


class PropertyListLayout(ContainerLayout):
    value = traitlets.List(help="List values").tag(sync=True)

    def __init__(self, value: typing.List, **kwargs):
        super().__init__(len(value), 2, value=value, **kwargs)
       
    def value_items(self, value):
        return enumerate(value)

    def entry_widget(self, kx, vx):
        tname = type(vx).__name__
        wdx = scalar_widget(tname, default=vx)
        wdx.add_traits(node_key=traitlets.Integer(kx))  # type: ignore
        wdx.observe(self.change_value)  # type: ignore
        return wdx

    def update_value(self, node_key: Integer, new_value) -> None:
        self.value[node_key] = new_value


class PropertyMapLayout(ContainerLayout):
    value = traitlets.Dict(help="Dict values").tag(sync=True)

    def __init__(self, value: typing.Dict, **kwargs):
        # print("Value: ", len(value))
        if len(value) > 0:
            super().__init__(len(value), 2, value=value, **kwargs)
        else:
            super().__init__(1, 1, value=value, **kwargs)

    def value_items(self, value) -> typing.Iterable:
        return value.items()

    def init_widget(self, vx) -> iwidgets.CoreWidget | None:
        tname = type(vx).__name__
        match tname:
            case "list":
                return PropertyListLayout(vx)
            case _:
                return scalar_widget(tname, default=vx)

    def entry_widget(self, kx, vx) -> iwidgets.CoreWidget | None:
        wdx = self.init_widget(vx)
        wdx.add_traits(node_key=traitlets.Unicode(kx))  # type: ignore
        wdx.observe(self.change_value)  # type: ignore
        return wdx


def container_layout(widget_key, **kwargs):
    match widget_key:
        case "dict":
            return PropertyMapLayout(value=kwargs["default"], **kwargs)
        case "list":
            return PropertyListLayout(value=kwargs["default"], **kwargs)


def render_property(widget_key, **kwargs):
    match widget_key:
        case "dict":
            return container_layout(widget_key, **kwargs)
        case _:
            return scalar_widget(widget_key, **kwargs)


#
# Base class for tree node
class CBTreeNode(itree.Node):
    node_key = traitlets.Unicode()


# Base class for side panel
class PanelBase:
    def __init__(self, **kwargs):
        self.layout = None
        self.widget_map = {}
        self.links = []

    def clear_links(self):
        for lx in self.links:
            lx.unlink()
        self.links.clear()

    def update(self, new_val):
        self.clear_links()
        if not new_val:
            return
        for slot, widx in self.widget_map.items():
            self.links.append(iwidgets.link((new_val, slot), (widx, "value")))


class CfgSidePanel(PanelBase):
    def __init__(self, template_map, **kwargs):
        super().__init__(**kwargs)
        for ekey, vmap in template_map[RecipeKeys.INIT_PARAMS].items():
            self.widget_map[ekey] = render_property(vmap[RecipeKeys.TYPE], **vmap)
        for ekey, vmap in template_map[RecipeKeys.EXEC_PARAMS].items():
            self.widget_map[ekey] = render_property(vmap[RecipeKeys.TYPE], **vmap)

        # Set Widgets
        ip_widgets: typing.List[iwidgets.Label | iwidgets.HBox] = []
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
        ep_widgets: typing.List[iwidgets.Label | iwidgets.HBox] = []
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
        self.layout = iwidgets.VBox(ip_widgets + ep_widgets)

    def wrap(self, widget_key: str, twid_def: typing.Dict[str, str]):
        # print("W Key", widget_key)
        return iwidgets.HBox(
            [
                iwidgets.Label(twid_def["label"] + " :"),
                self.widget_map[widget_key],
            ]
        )


class StructSidePanel(PanelBase):
    def __init__(self, struct_comp: structure.StructBase, **kwargs):
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
        self.layout = iwidgets.VBox(struct_widgets)

    def wrap(self, widget_key, twid_dict):
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
        **kwargs,
    ):
        self.tree: itree.Tree = itree.Tree()
        self.layout: iwidgets.TwoByTwoLayout = iwidgets.TwoByTwoLayout()
        self.panel_keys: typing.Set[str] = set([])
        self.panel_dict: typing.Dict[str, PanelBase] = {}
        self.left_width: str = left_width

    @abc.abstractmethod
    def init_tree(self) -> itree.Tree:
        return itree.Tree()

    @abc.abstractmethod
    def build(self) -> "TreeBase | None":
        return None

    def tree_update(self, change):
        new_val = change["new"]
        if not len(new_val):
            return
        _log().warning("Key : " + new_val[0].node_key)
        node_key = new_val[0].node_key
        if node_key in self.panel_dict:
            _log().warning("Key in Panel ")
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
        register_key: str, init_params: typing.Dict
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
        wf_step: typing.Dict[str, typing.Any]
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
        return ConfigTreeBase.type_trait_tree_node(step_key, wf_dict)


class ConfigTreeBase(TreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str,
        **kwargs,
    ):
        super().__init__(left_width, **kwargs)
        self.mdr_setup: RecipeSetup = mdr_setup

    def init_db_node(
        self, db_key: str, db_desc: typing.Dict[str, typing.Any]
    ) -> CBTreeNode:
        db_node = CBTreeNode(name=db_desc[RecipeKeys.LABEL], node_key=db_key)
        for wf_step in db_desc[RecipeKeys.WORKFLOW]:
            db_node.add_node(ConfigTreeBase.wflow_step_tree_node(wf_step))
            self.panel_keys.add(wf_step[RecipeKeys.NAME])
        return db_node

    def init_side_panels(self) -> typing.Dict[str, PanelBase]:
        _log().info(
            "Start Left-side panel construction for [%s]", str(self.panel_dict.keys())
        )
        for pkey in self.panel_keys:
            _log().debug("Initializing Panels for [%s]", pkey)
            ptemplate = self.mdr_setup.get_template_for(pkey)
            self.panel_dict[pkey] = CfgSidePanel(ptemplate)
        _log().info("Completed Left-side panel construction")
        return self.panel_dict

    def build(self) -> TreeBase:
        self.tree = self.init_tree()
        self.panel_dict = self.init_side_panels()
        self.tree.observe(self.tree_update, names="selected_nodes")  # type: ignore
        self.layout = iwidgets.TwoByTwoLayout(top_left=self.tree, bottom_right=None)
        return self

    @abc.abstractmethod
    def init_tree(self) -> itree.Tree:
        return itree.Tree()


class SourceDataTreeView(ConfigTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width="40%",
        **kwargs,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.src_data_desc = mdr_setup.recipe_sections[RecipeKeys.SRC_DATA]

    def init_tree(self) -> itree.Tree:
        root_node = CBTreeNode(name=CfgTreeNames.SRC_DATA, node_key="root")
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
        self.tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type:ignore
        return self.tree


class D2MLocationsTreeView(ConfigTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width="40%",
        **kwargs,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.d2m_map_desc = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]

    def init_neuron_node(
        self, neuron_name: str, neuron_desc: typing.Dict[str, typing.Any]
    ) -> CBTreeNode:
        neuron_node = CBTreeNode(name=neuron_name, node_key="d2m_map.neuron")
        for db_key, db_desc in neuron_desc[RecipeKeys.SRC_DATA].items():
            neuron_node.add_node(self.init_db_node(db_key, db_desc))
        return neuron_node

    def init_tree(self) -> itree.Tree:
        root_node = CBTreeNode(name=RecipeKeys.LOCATIONS, node_key="root")
        for loc_name, loc_desc in self.d2m_map_desc[RecipeKeys.LOCATIONS].items():
            loc_node = CBTreeNode(name=loc_name, node_key="d2m_map.location")
            for neuron_name, neuron_desc in loc_desc.items():
                neuron_node = self.init_neuron_node(neuron_name, neuron_desc)
                loc_node.add_node(neuron_node)
            root_node.add_node(loc_node)
        self.tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type: ignore
        return self.tree


class D2MConnectionsTreeView(ConfigTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width="40%",
        **kwargs,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.d2m_map_desc = mdr_setup.recipe_sections[RecipeKeys.DB2MODEL_MAP]

    def init_conncection_node(
        self, connect_name: str, connect_desc: typing.Dict[str, typing.Any]
    ) -> CBTreeNode:
        conn_node = CBTreeNode(name=connect_name, node_key="d2m_map.connection")
        for db_key, db_desc in connect_desc[RecipeKeys.SRC_DATA].items():
            db_node = CBTreeNode(name=db_desc[RecipeKeys.LABEL], node_key=db_key)
            for wf_step in db_desc[RecipeKeys.WORKFLOW]:
                db_node.add_node(ConfigTreeBase.wflow_step_tree_node(wf_step))
            self.panel_keys.union(
                set(
                    wf_step[RecipeKeys.NAME] for wf_step in db_desc[RecipeKeys.WORKFLOW]
                )
            )
            conn_node.add_node(self.init_db_node(db_key, db_desc))
        return conn_node

    def init_tree(self) -> itree.Tree:
        root_node = CBTreeNode(name=CfgTreeNames.CONNECTIONS, node_key="tree")
        for conn_name, conn_desc in self.d2m_map_desc[RecipeKeys.CONNECTIONS].items():
            root_node.add_node(self.init_conncection_node(conn_name, conn_desc))
        self.tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type: ignore
        return self.tree


class NetworkTreeView(TreeBase):
    def __init__(
        self,
        net_struct: structure.Network,
        left_width="40%",
        **kwargs,
    ) -> None:
        super().__init__(left_width, **kwargs)
        self.net_struct = net_struct

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

    def init_ext_net_side_panels(self, ext_net: structure.ExtNetwork) -> None:
        self.panel_dict[ext_net.name] = StructSidePanel(ext_net)
        for _, net_region in ext_net.locations.items():
            self.init_region_side_panels(net_region)
        for _, net_connect in ext_net.connections.items():
            self.init_conn_side_panels(net_connect)

    def init_data_link_panel(self, data_link: structure.DataLink) -> None:
        if data_link.name and data_link.property_map:
            self.panel_dict[data_link.name] = StructSidePanel(data_link)

    def init_data_file_panel(self, data_file: structure.DataFile) -> None:
        self.panel_dict[data_file.name] = StructSidePanel(data_file)

    def init_side_panels(self) -> typing.Dict[str, PanelBase]:
        _log().info(
            "Start Left-side panel construction for [%s]", str(self.net_struct.name)
        )
        self.panel_dict[self.net_struct.name] = StructSidePanel(self.net_struct)
        #
        for _, net_region in self.net_struct.locations.items():
            self.init_region_side_panels(net_region)
        #
        for _, net_connect in self.net_struct.connections.items():
            self.init_conn_side_panels(net_connect)
        #
        for _, ext_net in self.net_struct.ext_networks.items():
            self.init_ext_net_side_panels(ext_net)
        for data_cx in self.net_struct.data_connect:
            self.init_data_link_panel(data_cx)
        for data_fx in self.net_struct.data_files:
            self.init_data_file_panel(data_fx)
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

    def init_tree(self) -> itree.Tree:
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
        self.tree = itree.Tree(multiple_selection=False)
        self.tree.add_node(root_node)
        self.tree.layout.width = self.left_width  # type: ignore
        return self.tree

    def build(self) -> TreeBase:
        self.tree = self.init_tree()
        self.panel_dict = self.init_side_panels()
        self.tree.observe(self.tree_update, names="selected_nodes")  # type:ignore
        self.layout = iwidgets.TwoByTwoLayout(top_left=self.tree, bottom_right=None)
        return self

    def values_dict(self):
        pass
