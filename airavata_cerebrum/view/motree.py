import abc
import itertools
import logging
import typing as t
from collections.abc import Iterable

import awitree
import marimo as mo
import traitlets
from marimo._plugins.ui._core.ui_element import UIElement
from typing_extensions import override

from ..model import structure as structure
from ..model.setup import RecipeKeys, RecipeLabels, RecipeSetup
from . import (PanelBase, TreeBase, CBTreeNode, RcpTreeNames, StructTreeNames,
               recipe_traits)


MoUIElement : t.TypeAlias = UIElement[t.Any, t.Any]
MoPanelBaseType : t.TypeAlias = PanelBase[mo.Html, MoUIElement]
MoTreeBase: t.TypeAlias = TreeBase[MoPanelBaseType]

def _log():
    return logging.getLogger(__name__)


def scalar_widget(
    widget_key: str,
    label: str = "",
    **kwargs: t.Any
) -> MoUIElement | None:
    match widget_key:
        case "int" | "int32" | "int64":
            return mo.ui.number(value=kwargs["default"], label=label)
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
    def __init__(self, value: list[t.Any], label: str = "", **kwargs: t.Any):
        super().__init__(
            [wx for wx in self.widgets(value, **kwargs) if wx is not None],
            label=label,
        )

    def widgets(self, value: list[t.Any], **_kwargs: t.Any):
        return (
            scalar_widget(type(vx).__name__, label=str(kx) + " :", default=vx)
            for kx, vx in enumerate(value)
        )


# NOTE: Features of trait setting and change handler not included
class PropertyMapLayout(mo.ui.dictionary):

    def __init__(
        self,
        value: dict[str, t.Any],
        label: str = "",
        **kwargs: t.Any
    ):
        super().__init__(
            {kx: wx for kx, wx in self.widgets(value, **kwargs) if wx is not None},
            label=label,
        )
        # print("Value: ", len(value))

    def init_widget(
        self, vx: t.Any, label: str = "", **_kwargs: t.Any
    ) -> MoUIElement | None:
        tname = type(vx).__name__
        match tname:
            case "NoneType":
                return scalar_widget("str", default="None", label=label)
            case "list":
                return PropertyListLayout(vx, label="")
            case _:
                return scalar_widget(tname, default=vx, label=label)

    def widgets(
        self, value: dict[str, t.Any], **kwargs: t.Any
    ) -> Iterable[tuple[str, MoUIElement | None]]:
        return ((kx, self.init_widget(vx, **kwargs)) for kx, vx in value.items())


def render_property(
    widget_key: str, label: str = "", **kwargs: t.Any
) -> MoUIElement | None:
    # _log().warning("Render Out [%s]", str(kwargs))
    match widget_key:
        case "dict":
            return PropertyMapLayout(value=kwargs["default"], label=label, **kwargs)
        case "list":
            return PropertyListLayout(value=kwargs["default"], **kwargs)
        case _:
            return scalar_widget(widget_key, label=label, **kwargs)


class DBWorkflowSidePanel(MoPanelBaseType):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        workfow_desc: list[dict[str, t.Any]],
        delay_build: bool = False,
        **kwargs: t.Any,
    ):
        super().__init__(**kwargs)
        self.mdr_setup: RecipeSetup = mdr_setup
        self.workfow_desc: list[dict[str, t.Any]] = workfow_desc
        # Setup Widgets
        if delay_build:
            self.set_layout(None)
        else:
            self.set_layout(self.build_layout())

    @override
    def build_layout(self) -> mo.Html | None:
        return mo.vstack(
            [
                mo.ui.array(
                    self.render_workflow_step(wf_step),
                    label=f"Step {wf_idx + 1} : {wf_step[RecipeKeys.LABEL]}",
                )
                for wf_idx, wf_step in enumerate(self.workfow_desc)
            ]
        )

    def render_workflow_step(
        self, wf_step: dict[str, t.Any]
    ) -> list[MoUIElement]:
        step_key = wf_step[RecipeKeys.NAME]
        _log().debug("Initializing Panels for [%s]", step_key)
        rcp_template = self.mdr_setup.get_template_for(step_key)
        _, rcp_traits = recipe_traits(wf_step)
        return self.workflow_ui(
            rcp_template,
            rcp_traits,
        )

    def workflow_ui(
        self,
        template_map: dict[str, t.Any],
        elt_traits: traitlets.HasTraits | None = None,
    ) -> list[MoUIElement]:
        trait_vals = elt_traits.trait_values() if elt_traits else {}
        return list(itertools.chain.from_iterable(
            (
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
            )
        ))

    def params_widget(
        self,
        template_map: dict[str, t.Any],
        trait_vals: dict[str, t.Any],
        params_key: str,
        params_label: str,
    ) -> Iterable[MoUIElement]:
        if template_map[params_key]:
            wd_itr: Iterable[UIElement[t.Any, t.Any] | None] = (
                self.property_widget(trait_vals, ekey, vmap)
                for ekey, vmap in template_map[params_key].items()
            )
            # wd_list = [wx for wx in wd_itr if wx is not None]
            # return wd_list
            return (wx for wx in wd_itr if wx is not None)
            # return mo.ui.array(
            #     [wx for wx in wd_itr if wx is not None],
            #     label=params_label
            # )
        else:
            return []
        # return mo.ui.array([], label=params_label)

    def property_widget(
        self,
        traitv_dct: dict[str, t.Any],
        ekey: str,
        vmap: dict[str, t.Any],
    ):
        if ekey in traitv_dct:
            vmap["default"] = traitv_dct[ekey]
        return render_property(vmap[RecipeKeys.TYPE], **vmap)


class StructSidePanel(MoPanelBaseType):
    def __init__(
        self,
        struct_comp: structure.StructBase,
        delay_build: bool = False,
        **kwargs: t.Any,
    ):
        super().__init__(**kwargs)
        self._struct: structure.StructBase = struct_comp
        if delay_build:
            self.set_layout(None)
        else:
            self.set_layout(self.build_layout())

    @override
    def build_layout(self):
        # Set Widgets
        wd_itr: Iterable[UIElement[t.Any, t.Any] | None] = (
            render_property(
                vmap.value_type,
                label=vmap.label + " :",
                default=vmap.to_ui(self._struct.get(ekey)),
            )
            for ekey, vmap in self._struct.trait_ui().items()
        )
        ui_elements: list[UIElement[t.Any, t.Any]] = [
            wx for wx in wd_itr if wx is not None
        ]
        return mo.vstack([mo.ui.array(ui_elements, label=self._struct.name)])


class RecipeTreeBase(TreeBase[MoPanelBaseType], metaclass=abc.ABCMeta):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: float = 0.4,
        **kwargs: t.Any,
    ):
        super().__init__(**kwargs)
        self.mdr_setup: RecipeSetup = mdr_setup
        self.root_node: CBTreeNode | None = None
        self.left_width: float = left_width
        self.widths: list[float] = [left_width, 1 - left_width]

    def db_workflow_recipe_node(
        self,
        db_key: str,
        db_desc: dict[str, t.Any],
        delay_build: bool = False,
    ) -> tuple[CBTreeNode, MoPanelBaseType]:
        db_node = CBTreeNode.init(name=db_desc[RecipeKeys.LABEL], key=db_key)
        _log().info("Left-side panel construction for [%s]", str(db_key))
        db_panel = DBWorkflowSidePanel(
            self.mdr_setup,
            db_desc[RecipeKeys.WORKFLOW],
            delay_build
        )
        _log().info("Left-side panel construction")
        return db_node, db_panel

    @override
    def set_layout(self, selected_panel: MoPanelBaseType) -> None:
        self.layout_ = mo.hstack([self.tree, selected_panel.layout])

    @override
    def build(self, root_pfx: str = "") -> TreeBase[MoPanelBaseType]:
        self.tree: awitree.Tree | None = self.build_tree(root_pfx)
        # self.tree.observe(self.tree_update, names="selected_nodes")
        self.layout_: mo.Html | None = mo.hstack(
            [self.tree, mo.vstack([])], widths=[self.left_width, 1 - self.left_width]
        )
        return self

    @override
    @abc.abstractmethod
    def build_tree(self, root_pfx: str = "") -> awitree.Tree:
        pass


class DataSourceRecipeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: float = 0.4,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
        self.src_data_desc: dict[str, t.Any] = mdr_setup.recipe_sections[
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
        # initialize tree
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened": True}
        tree: awitree.Tree = awitree.Tree(data=tree_data)
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree


class Data2ModelRecipeView(RecipeTreeBase):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        left_width: float = 0.4,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(mdr_setup, left_width, **kwargs)
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
        neuron_node = CBTreeNode.init(name=neuron_name, key=f"d2m_map-{neuron_pfx}")
        for db_key, db_desc in neuron_desc[RecipeKeys.SRC_DATA].items():
            db_loc_key = f"{neuron_pfx}-{db_key}"
            db_node, db_panel = self.db_workflow_recipe_node(db_loc_key, db_desc)
            self.panel_dict[db_loc_key] = db_panel
            neuron_node.add_node(db_node)
        return neuron_node

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

    def conncection_node(
        self, connect_name: str, connect_desc: dict[str, t.Any], conn_node_pfx: str
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

    @override
    def build_tree(self, root_pfx: str = "") -> awitree.Tree:
        root_key = f"{root_pfx}-{RecipeKeys.DB2MODEL_MAP}"
        self.root_node: CBTreeNode | None = CBTreeNode.init(
            name=RcpTreeNames.D2M, key=root_key
        )
        self.root_node.add_node(self.locations(root_key))
        self.root_node.add_node(self.connections(root_key))
        # initialize tree
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened": True}
        tree: awitree.Tree = awitree.Tree(data=tree_data)
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree


class NetworkStructureView(TreeBase[MoPanelBaseType]):
    def __init__(
        self,
        net_struct: structure.Network,
        left_width: float = 0.4,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(**kwargs)
        self.net_struct: structure.Network = net_struct
        self.left_width: float = left_width
        self.widths: list[float] = [left_width, 1 - left_width]

    @override
    def set_layout(self, selected_panel: MoPanelBaseType) -> None:
        self.layout_ = mo.hstack([self.tree, selected_panel.layout])

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
    def build(self, root_pfx: str = "") -> TreeBase[MoPanelBaseType]:
        self.tree: awitree.Tree | None = self.build_tree(root_pfx)
        # self.panel_dict : dict[str, PanelBase] = self.build_side_panels()
        # self.tree.observe(self.tree_update, names="selected_nodes")  # type:ignore
        self.layout_: mo.Html | None = mo.hstack(
            [self.tree, mo.vstack([])], widths=self.widths
        )
        return self

    def values_dict(self):
        pass


class RecipeExplorer(TreeBase[MoPanelBaseType]):
    def __init__(
        self,
        mdr_setup: RecipeSetup,
        net_struct: structure.Network,
        left_width: float = 0.4,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(**kwargs)
        self._data_src: DataSourceRecipeView = DataSourceRecipeView(
            mdr_setup, left_width, **kwargs
        )
        self._d2m: Data2ModelRecipeView = Data2ModelRecipeView(
            mdr_setup, left_width, **kwargs
        )
        self._struct: NetworkStructureView = NetworkStructureView(
            net_struct, left_width, **kwargs
        )
        self.left_width: float = left_width
        self.widths: list[float] = [left_width, 1 - left_width]

    @override
    def set_layout(self, selected_panel: MoPanelBaseType) -> None:
        self.layout_ = mo.hstack([self.tree, selected_panel])

    def build_tree(self, root_pfx: str = "") -> awitree.Tree:
        root_key = f"{root_pfx}-integ"
        self._data_src.build(root_key)
        self._d2m.build(root_key)
        self._struct.build(root_key)
        # initialize tree
        self.root_node: CBTreeNode | None = CBTreeNode.init(
            name=f"{root_pfx}-{RcpTreeNames.RECIPE}",
            key=root_key
        )
        if self._data_src.root_node:
            self.root_node.add_node(self._data_src.root_node)
        if self._d2m.root_node:
            self.root_node.add_node(self._d2m.root_node)
        if self._struct.root_node:
            self.root_node.add_node(self._struct.root_node)
        # build tree data
        tree_data = self.root_node.awi_dict()
        tree_data["state"] = {"selected": True, "opened": True}
        tree: awitree.Tree = awitree.Tree(data=tree_data)
        # self.tree.layout.width = self.left_width  # type:ignore
        return tree

    @override
    def build(self, root_pfx: str = "") -> TreeBase[MoPanelBaseType]:
        self.tree: awitree.Tree | None = self.build_tree(root_pfx)
        # self.panel_dict : dict[str, PanelBase] = self.build_side_panels()
        # self.tree.observe(self.tree_update, names="selected_nodes")  # type:ignore
        self.layout_: mo.Html | None = mo.hstack(
            [self.tree, mo.vstack([])], widths=self.widths
        )
        return self

    @override
    def view_components(self):
        return (
            self.tree,
            (
                self._data_src.panel_dict,
                self._d2m.panel_dict,
                self._struct.panel_dict,
            ),
        )
