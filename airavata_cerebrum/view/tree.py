import abc
import itertools
import logging
import typing as t
#
import awitree
import ipywidgets as iwidgets
from pydantic.fields import FieldInfo
import traitlets

#
from collections.abc import Iterable
from typing_extensions import override

from ..base import CerebrumBaseModel, BaseStruct, BaseParams, INPGT, EXPGT
from ..model.setup import RecipeKeys, RecipeSetup
from ..model import structure as structure
from . import (RcpTreeNames, StructTreeNames, workflow_params,
               BaseTree, CBTreeNode, BasePanel)

IPyPanelT : t.TypeAlias = BasePanel[iwidgets.CoreWidget, iwidgets.Box]

def _log():
    return logging.getLogger(__name__)


def scalar_widget(
    widget_key: str,
    value: t.Any,
    **kwargs: t.Any
) -> iwidgets.CoreWidget | None:
    match widget_key:
        case "int" | "int32" | "int64":
            return iwidgets.IntText(value=value, disabled=False)
        case "float" | "float32" | "float64":
            return iwidgets.FloatText(value=value, disabled=False)
        case "text":
            return iwidgets.Text(value=value, disabled=False)
        case "textarea" | "str":
            return iwidgets.Textarea(value=value, disabled=False)
        case "option":
            return iwidgets.Dropdown(
                disabled=False,
                value=value,
                options=kwargs["options"],
            )
        case "check" | "bool":
            return iwidgets.Checkbox(
                value=bool(value),
                disabled=False,
                indent=True,
            )
        case "tags":
            return iwidgets.TagsInput(
                value=value,
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
            wdx = scalar_widget(type(vx).__name__, vx)
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

class PropertyTupleLayout(iwidgets.GridspecLayout):
    value : traitlets.Tuple = traitlets.Tuple(
        help="Tuple values"
    ).tag(sync=True)

    def __init__(self, value: tuple[t.Any], **kwargs: t.Any):
        super().__init__(len(value), 2, value=value, **kwargs)
        for ix, (kx, vx) in enumerate(enumerate(value)):
            wdx = scalar_widget(type(vx).__name__, vx)
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
            tlist = list(self.value)
            tlist[entry_index] = new_value
            self.value = tuple(tlist)


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
                return scalar_widget("str", "None")
            case "list":
                return PropertyListLayout(vx)
            case "tuple":
                return PropertyTupleLayout(vx)
            case _:
                return scalar_widget(tname, vx)


def render_property(
    widget_key: str,
    value: t.Any,
    **kwargs: t.Any
) -> iwidgets.CoreWidget | None:
    match widget_key:
        case "dict":
            return PropertyMapLayout(value=value, **kwargs)
        case "list":
            return PropertyListLayout(value=value, **kwargs)
        case "tuple":
            return PropertyTupleLayout(value=value, **kwargs)
        case _:
            return scalar_widget(widget_key, value, **kwargs)


# Base class for side panel
class DBWorkflowSidePanel(IPyPanelT):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        workflow_desc: list[dict[str, t.Any]],
        delay_build: bool = False,
        **kwargs: t.Any,
    ):
        super().__init__(**kwargs)
        self.mdr_setup: RecipeSetup = mdr_setup
        self.workflow_desc: list[dict[str, t.Any]] = workflow_desc
        # Setup Widgets
        if delay_build:
            self.set_layout(None)
        else:
            self.set_layout(self.build_layout())

    @override
    def build_layout(self) -> iwidgets.Box | None:
        return iwidgets.VBox(
            [
                iwidgets.VBox(
                    [
                        iwidgets.Label(
                            f"Step {wf_idx + 1} : {wf_step[RecipeKeys.LABEL]} ::",
                            style={"font_weight": "bold"},
                        )
                    ] +
                    self.render_workflow_step(wf_step),
                    layout=iwidgets.Layout(border="solid")
                )
                for wf_idx, wf_step in enumerate(self.workflow_desc)
            ]
        )

    def render_workflow_step(
        self, wf_step: dict[str, t.Any]
    ) -> list[iwidgets.CoreWidget]:
        step_key = wf_step[RecipeKeys.NAME]
        # _log().warning("Initializing Panels for [%s]", step_key)
        rcp_template = self.mdr_setup.get_template_for(step_key)
        _, rcp_traits = workflow_params(wf_step)
        return self.workflow_ui(
            rcp_template,
            rcp_traits,
        )

    def workflow_ui(
        self,
        rcp_template: dict[str, t.Any],
        wf_params: BaseParams[INPGT, EXPGT]  | None = None,
    ) -> list[iwidgets.CoreWidget]:
        return [
            self.workflow_params_widget(
                rcp_template[RecipeKeys.INIT_PARAMS],
                wf_params.init_params if wf_params else None,
            ),
            self.workflow_params_widget(
                rcp_template[RecipeKeys.EXEC_PARAMS],
                wf_params.exec_params if wf_params else None,
            ),
        ]

    def render_widget(
        self,
        widget_key: str,
        value: t.Any,
        **kwargs: t.Any):
        return iwidgets.HBox(
            [
                iwidgets.Label(kwargs[RecipeKeys.LABEL] + " :"),
                render_property(widget_key, value, **kwargs,),
            ]
        )

    def workflow_params_widget(
        self,
        wf_template: dict[str, t.Any] | None,
        wf_params: CerebrumBaseModel | None ,
    ) -> iwidgets.CoreWidget:
        wd_itr: Iterable[iwidgets.CoreWidget | None] = ()
        if wf_template and wf_params:
            wd_itr = (
                self.render_widget(
                    tdesc[RecipeKeys.TYPE],
                    wf_params.get(tkey, tdesc["default"]),
                    **tdesc  
                )
                for tkey, tdesc in wf_template.items()
                if tkey not in wf_params.exclude()
            )
        elif wf_template:
            wd_itr = (
                self.render_widget(
                    tdesc[RecipeKeys.TYPE],
                    tdesc["default"],
                    **tdesc
                )
                for _tkey, tdesc in wf_template.items()
            )
        elif wf_params:
            wd_itr = (
                self.render_widget(
                     tdesc.annotation.__name__,
                     value = wf_params.get(tkey),
                     label = str(tdesc.title),
                     options = [wf_params.get(tkey)],
                     allowed = [wf_params.get(tkey)]
                )
                for tkey, tdesc in wf_params.model_fields.items()
                if tkey not in wf_params.exclude()
            )

        return iwidgets.VBox(list(wd_itr))



class StructSidePanel(IPyPanelT):
    def __init__(
        self,
        struct_comp: BaseStruct,
        delay_build: bool = False,
        **kwargs: t.Any,
    ):
        super().__init__(**kwargs)
        self._struct: BaseStruct = struct_comp
        if delay_build:
            self.set_layout(None)
        else:
            self.set_layout(self.build_layout())

    def wrap_hbox(
            self,
            field_info: FieldInfo,
            rwidget: iwidgets.CoreWidget | None
    ) -> iwidgets.HBox | None:
        if rwidget is not None:
            return iwidgets.HBox([
                iwidgets.Label(str(field_info.title) + " :"),
                rwidget
            ])
        else:
            return None

    @override
    def build_layout(self):
        # Set Widgets
        wd_itr: Iterable[iwidgets.HBox | None] = (
            self.wrap_hbox(
                field_info,
                render_property(
                    field_info.annotation.__name__,
                    self._struct.get(fkey),
                )
            )
            for fkey, field_info in self._struct.model_fields.items()
            if fkey not in self._struct.exclude()
        )
        ui_elements: list[iwidgets.CoreWidget] = [
            wx for wx in wd_itr if wx is not None
        ]
        return iwidgets.VBox(
            [iwidgets.Label(self._struct.name + " : ")] + ui_elements
        )


class RecipeTreeBase(BaseTree[IPyPanelT], metaclass=abc.ABCMeta):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        right_width: str,
        **kwargs: t.Any,
    ):
        super().__init__(**kwargs)
        self.mdr_setup: RecipeSetup = mdr_setup
        self.root_node: CBTreeNode | None = None
        self.right_width : str = right_width

    def db_workflow_recipe_node(
        self,
        db_key: str,
        db_desc: dict[str, t.Any],
        delay_build: bool = False,
    ) -> tuple[CBTreeNode, IPyPanelT]:
        db_node = CBTreeNode.init(name=db_desc[RecipeKeys.LABEL], key=db_key)
        _log().info("Start Left-side panel construction for [%s]", str(db_key))
        db_panel = DBWorkflowSidePanel(
            self.mdr_setup, db_desc[RecipeKeys.WORKFLOW], delay_build
        )
        _log().info("Completed Left-side panel construction")
        return db_node, db_panel

    @override
    def build(self, root_pfx: str = "") -> BaseTree[IPyPanelT]:
        self.tree: awitree.Tree | None = self.build_tree(root_pfx)
        self.tree.observe(self.tree_update, names="selected_nodes")
        self.layout: iwidgets.TwoByTwoLayout = iwidgets.TwoByTwoLayout(
            top_left=self.tree,
            bottom_right=None
        )
        return self

    @override
    def set_layout(self, selected_panel: IPyPanelT) -> None:
          self.layout.bottom_right = selected_panel.layout

    @override
    @abc.abstractmethod
    def build_tree(self, root_pfx: str = "") -> awitree.Tree:
        pass


class DataSourceRecipeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.src_data_desc : dict[str, t.Any] = mdr_setup.recipe_sections[
            RecipeKeys.SRC_DATA
        ]

    @override
    def build_tree(self, root_pfx: str = "") -> awitree.Tree:
        root_key = f"{root_pfx}-{RecipeKeys.SRC_DATA}"
        self.root_node: CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.SRC_DATA, key=root_key
        )
        for db_key, db_desc in self.src_data_desc.items():
           src_db_key = f"{root_key}-{db_key}"
           wf_iter = itertools.chain(
                db_desc[RecipeKeys.DB_CONNECT][RecipeKeys.WORKFLOW],
                db_desc[RecipeKeys.POST_OPS][RecipeKeys.WORKFLOW],
            )
           db_node, db_panel = self.db_workflow_recipe_node(
                src_db_key,
                {
                    RecipeKeys.LABEL: db_desc[RecipeKeys.LABEL],
                    RecipeKeys.WORKFLOW: wf_iter,
                },
           )
           self.panel_dict[src_db_key] = db_panel
           self.root_node.add_node(db_node)
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened": True}
        tree: awitree.Tree = awitree.Tree(data=tree_data)
        #self.layout_.width = self.left_width
        return tree


class Data2ModelRecipeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        right_width: str="60%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, right_width, **kwargs)
        self.d2m_map_desc: dict[str, t.Any] = mdr_setup.recipe_sections[
            RecipeKeys.DB2MODEL_MAP
        ]

    def neuron_node(
        self,
        neuron_name: str,
        neuron_desc: dict[str, t.Any],
        loc_node_pfx: str
    ) -> CBTreeNode:
        neuron_pfx: str = f"{loc_node_pfx}-{neuron_name}"
        neuron_node = CBTreeNode.init(
            name=neuron_name,
            key=f"{RcpTreeNames.D2M_NEURON}-{neuron_pfx}"
        )
        for db_key, db_desc in neuron_desc[RecipeKeys.SRC_DATA].items():
            db_loc_key = f"{neuron_pfx}-{db_key}"
            db_node, db_panel = self.db_workflow_recipe_node(db_loc_key, db_desc)
            self.panel_dict[db_loc_key] = db_panel
            neuron_node.add_node(db_node)
        return neuron_node

    def conncection_node(
        self,
        connect_name: str,
        connect_desc: dict[str, t.Any],
        conn_node_pfx: str
    ) -> CBTreeNode:
        conn_node_key = f"{conn_node_pfx}-{connect_name}"
        conn_node = CBTreeNode.init(name=connect_name, key=conn_node_key)
        for db_key, db_desc in connect_desc[RecipeKeys.SRC_DATA].items():
            db_conn_key = f"{conn_node_key}-{db_key}"
            db_node, db_panel = self.db_workflow_recipe_node(db_conn_key, db_desc)
            self.panel_dict[db_conn_key] = db_panel
            conn_node.add_node(db_node)
        return conn_node

    def connections(self, conn_pfx: str):
        conn_key = f"{conn_pfx}-{RecipeKeys.CONNECTIONS}"
        root_conn_node: CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.CONNECTIONS, key=conn_key
        )
        for conn_name, conn_desc in self.d2m_map_desc[RecipeKeys.CONNECTIONS].items():
            root_conn_node.add_node(
                self.conncection_node(conn_name, conn_desc, conn_key)
            )
        return root_conn_node

    def locations(self, loc_pfx: str = ""):
        loc_key = f"{loc_pfx}-{RecipeKeys.LOCATIONS}"
        loc_root_node: CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.LOCATIONS, key=loc_key
        )
        for loc_name, loc_desc in self.d2m_map_desc[RecipeKeys.LOCATIONS].items():
            loc_node_key = f"{loc_key}-{loc_name}"
            loc_node = CBTreeNode.init(name=loc_name, key=loc_node_key)
            for neuron_name, neuron_desc in loc_desc.items():
                neuron_node = self.neuron_node(neuron_name, neuron_desc, loc_node_key)
                loc_node.add_node(neuron_node)
            loc_root_node.add_node(loc_node)
        return loc_root_node

    @override
    def build_tree(self, root_pfx: str = "") -> awitree.Tree:
        root_key = f"{root_pfx}-{RecipeKeys.DB2MODEL_MAP}"
        # tree root node
        self.root_node: CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.D2M, key=root_key
        )
        self.root_node.add_node(self.locations(root_key))
        self.root_node.add_node(self.connections(root_key))
        # initialize tree
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened": True}
        tree: awitree.Tree = awitree.Tree(data=tree_data)
        #self.layout_.width = self.left_width
        return tree


class NetworkStructureView(BaseTree[IPyPanelT]):
    def __init__(
        self,
        net_struct: structure.Network,
        left_width: str="40%",
        **kwargs: t.Any,
    ) -> None:
        super().__init__(**kwargs)
        self.net_struct : structure.Network = net_struct
        self.left_width : str = left_width

    @override
    def set_layout(self, selected_panel: IPyPanelT) -> None:
          self.layout.bottom_right = selected_panel.layout

    def region_node(
        self, net_region: structure.Region, region_pfx: str = "r"
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
                neuron_node.add_node(CBTreeNode.from_struct(nx_model, model_key))
                self.panel_dict[model_key] = StructSidePanel(nx_model, True)
            region_node.add_node(neuron_node)
        return region_node

    def connection_node(
        self, net_connect: structure.Connection, connect_pfx: str = "c"
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
        self, ext_net: structure.ExtNetwork, ext_prefix: str = ""
    ) -> CBTreeNode:
        ext_key = f"{ext_prefix}-{ext_net.name}"
        ext_net_node = CBTreeNode.from_struct(ext_net, ext_key)
        self.panel_dict[ext_key] = StructSidePanel(ext_net)
        location_node = CBTreeNode.init(
            StructTreeNames.REGIONS, ext_net.name + ".locations"
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
        self, data_link: structure.DataLink, dl_prefix: str = ""
    ) -> CBTreeNode | None:
        if data_link.name and data_link.property_map:
            dl_key = f"{dl_prefix}-{data_link.name}"
            dl_node = CBTreeNode.from_struct(data_link, dl_key)
            self.panel_dict[dl_key] = StructSidePanel(data_link)
            return dl_node
        return None

    def data_file_node(
        self, data_file: structure.DataFile, df_prefix: str = ""
    ) -> CBTreeNode | None:
        df_key = f"{df_prefix}-{data_file.name}"
        df_node = CBTreeNode.from_struct(data_file, df_key)
        self.panel_dict[df_key] = StructSidePanel(data_file)
        return df_node

    def locations(self, root_key: str):
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
        ext_net_node = CBTreeNode.init(StructTreeNames.EXTERNAL_NETWORKS, ext_key)
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

    def build_tree_nodes(self, root_pfx: str) -> CBTreeNode:
        root_key = f"{root_pfx}-{self.net_struct.name}"
        #
        root_node = CBTreeNode.from_struct(self.net_struct, root_key)
        root_node.add_node(self.locations(root_key))
        root_node.add_node(self.connections(root_key))
        root_node.add_node(self.ext_networks(root_key))
        root_node.add_node(self.data_links(root_key))
        root_node.add_node(self.data_files(root_key))
        return root_node

    def build_tree(self, root_pfx: str = "") -> awitree.Tree:
        self.root_node: CBTreeNode | None = self.build_tree_nodes(root_pfx)
        # initialize tree
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened": True}
        tree: awitree.Tree = awitree.Tree(data=tree_data)
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree

    @override
    def build(self, root_pfx: str = "") -> BaseTree[IPyPanelT]:
        self.tree: awitree.Tree | None = self.build_tree(root_pfx)
        self.tree.observe(self.tree_update, names="selected_nodes")  # type:ignore
        self.layout : iwidgets.TwoByTwoLayout = iwidgets.TwoByTwoLayout(
            top_left=self.tree,
            bottom_right=None,
        )
        return self

    def values_dict(self):
        pass
