import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import airavata_cerebrum.model.setup as cbm_setup
    import airavata_cerebrum.model.recipe as cbm_recipe
    import airavata_cerebrum.view.motree as cbm_motree
    import airavata_cerebrum.model.structure as cbm_structure
    import airavata_cerebrum.util as cbm_utils
    import mousev1.model as v1model
    import marimo as mo
    return (
        cbm_motree,
        cbm_recipe,
        cbm_setup,
        cbm_structure,
        cbm_utils,
        json,
        mo,
        v1model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Recipe for a data-driven model
        A data-driven Model consists of three parts

        1. Source Data : Data Providers and the workflow to access data.
        2. Data to Model Mapping: A Mapping Function to map that funnels the appropriate data to different parts of the model.
        3. Custom Modifications: Description of the model that fills in the details not found in the dataset.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Model Recipe

        Model recipe is defined by a set of recipe files and template files. Recipe and Template files are described in __json__ files, and are expected to be placed in the __Recipe Directory__, which is canonically under the base directory of the model.

        The Recipe contains two sections:

        - Source Data
        - Data2Model Section 

        **Source Data** section describes how the source databases are connected and different operations such as filters are applied to the data available from the specific database.

        **Data2Model** section describes how the different parts of the model map to the source data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Source Data Section 

        Data provider module includes:

          - Methods for defining the querying from the database and filter the data based on specific criteria. 
          - Utilities to visualize the data provider configurations in easy-to-understand explorer view inside Jupyter/marimo notebook with the relevant parameters displayed in the side panel.

        Construction of Mouse V1 is shown below with three different data providers: 

          - Allen Cell Types database,
          - Allen Brain Cell Atlas and
          - AI Synaptic Physiology Database.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Data2Model Map

        For the V1 model, definitions for data2model includes two parts:

        1. *Locations:* Cell types database and the MERFISH atlas data map to neuron models and the distribution of neuron types, respectively
        2. *Connections:* AI synaptic physiology data is mapped to the connection probabilities between the pairs of neuron classes.
        """
    )
    return


@app.cell
def _():
    m_name = "v1"
    m_base_dir = "./"
    m_rcp_dir = "./v1/recipe/"
    m_rcp_files = {
        "recipe": [
            "recipe.json",
            "recipe_data.json",
            "recipe_dm_l1.json",
            "recipe_dm_l23.json",
            "recipe_dm_l4.json"
        ],
        "templates": [
            "recipe_template.json"
        ]
    }
    cmod_files = [
        "./v1/recipe/custom_mod.json",
        "./v1/recipe/custom_mod_l1.json",
        "./v1/recipe/custom_mod_l23.json",
        "./v1/recipe/custom_mod_l4.json",
        "./v1/recipe/custom_mod_ext.json",
        "./v1/recipe/custom_mod_ext_lgn.json",
        "./v1/recipe/custom_mod_ext_bkg.json",
    ]
    return cmod_files, m_base_dir, m_name, m_rcp_dir, m_rcp_files


@app.cell
def _(
    cbm_setup,
    cbm_structure,
    cmod_files,
    m_base_dir,
    m_name,
    m_rcp_dir,
    m_rcp_files,
):
    # recipe_dict = cbm_utils.io.load_json(main_rcp_file)
    # cmod_dict = cbm_utils.io.load_json(custom_mod_file)
    mdr_setup = cbm_setup.RecipeSetup(
        name=m_name,
        base_dir=m_base_dir,
        recipe_dir=m_rcp_dir,
        recipe_files=m_rcp_files,
    )

    tree_view_widths = [0.4, 0.6]
    cmod_struct = cbm_structure.Network.from_file_list(cmod_files)
    return cmod_struct, mdr_setup, tree_view_widths


@app.cell
def _(cbm_structure):
    tx = cbm_structure.Network.from_file_list( [
        "./v1/recipe/custom_mod.json",
        "./v1/recipe/custom_mod_l1.json",
        "./v1/recipe/custom_mod_l23.json",
        "./v1/recipe/custom_mod_l4.json",
        "./v1/recipe/custom_mod_ext.json",
        "./v1/recipe/custom_mod_ext_lgn.json",
        "./v1/recipe/custom_mod_ext_bkg.json",
    ])
    tx.ext_networks['bkg']
    return (tx,)


@app.cell
def _(cbm_motree, cmod_struct, mdr_setup, mo):
    #
    integ_explorer = cbm_motree.RecipeExplorer(mdr_setup, cmod_struct).build(root_pfx="V1")
    integ_tree,integ_panels = integ_explorer.view_components()
    integ_motree = mo.ui.anywidget(integ_tree)
    return integ_explorer, integ_motree, integ_panels, integ_tree


@app.cell(hide_code=True)
def _(cbm_motree, integ_motree, integ_panels, mo, tree_view_widths):
    #
    integ_selected = cbm_motree.BaseTree.panel_selector(integ_motree, integ_panels)
    mo.hstack(
        [
            integ_motree,
            integ_selected.layout if integ_selected else None
        ],
        widths=tree_view_widths
    )
    return (integ_selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Downloaded data as a database

        The downloaded data by default is stored as json file. Optionally, it can be saved to a duckdb database, which can be queried within the notebook.
        """
    )
    return


@app.cell
def _(cmod_struct):
    cmod_struct
    return


@app.cell
def _(cbm_recipe, cmod_struct, mdr_setup, v1model):
    #
    mdrecipe = cbm_recipe.ModelRecipe(
        recipe_setup=mdr_setup,
        region_mapper=v1model.V1RegionMapper,
        neuron_mapper=v1model.V1NeuronMapper,
        connection_mapper=v1model.V1ConnectionMapper,
        network_builder=v1model.V1BMTKNetworkBuilder,
        mod_structure=cmod_struct,
        save_flag=True,
    )
    #
    #
    # mdrecipe.acquire_source_data()
    return (mdrecipe,)


@app.cell
def _():
    import duckdb

    db_conn = duckdb.connect("./v1/recipe/db_connect_output.db")
    return db_conn, duckdb


@app.cell
def _(abm_mouse, db_conn, mo):
    abm_mouse_df = mo.sql(
        f"""
        SELECT * FROM abm_mouse LIMIT 100
        """,
        engine=db_conn
    )
    return (abm_mouse_df,)


@app.cell
def _(abm_celltypes_ct, db_conn, mo):
    import quak
    cols = ["cell_reporter_status", "line_name",
            "structure__layer",
           "donor__sex", "donor__age"]
    abm_ct_df = mo.sql(
        f"""
        SELECT {",".join(cols)} FROM abm_celltypes_ct
        """,
        engine=db_conn
    )
    quak.Widget(abm_ct_df)
    return abm_ct_df, cols, quak


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Map Data, Apply Mods, and Build Network Representation

        After the soruce data is acquired, the data is mapped to the locations and connection to create an intermediate representation.
        """
    )
    return


@app.cell
def _(cbm_motree, mdrecipe, mo):
    mdrecipe.source_data2model_struct()
    mdrecipe.apply_mod()
    final_net_view = cbm_motree.NetworkStructureView(mdrecipe.network_struct).build("Final-V1")

    fnet_tree,fnet_panels = final_net_view.view_components()
    fnet_motree = mo.ui.anywidget(fnet_tree)
    #final_net_view.tree
    return final_net_view, fnet_motree, fnet_panels, fnet_tree


@app.cell(hide_code=True)
def _(cbm_motree, fnet_motree, fnet_panels, mo, tree_view_widths):
    #
    fnet_selected = cbm_motree.BaseTree.panel_selector(fnet_motree, fnet_panels)
    mo.hstack(
        [
            fnet_motree,
            fnet_selected.layout if fnet_selected else None
        ],
        widths=tree_view_widths
    )
    return (fnet_selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Generate Network using BMTK

        Using the following helper classes, the above structure is translated into SONATA file:

        - v1model.V1RegionMapper
        - v1model.V1NeuronMapper
        - v1model.V1ConnectionMapper
        - v1model.V1BMTKNetworkBuilder
        """
    )
    return


@app.cell
def _():
    #
    #
    # mdrecipe.build_network()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Run with Nest

        Finally, Simulation can be run using nest as below
        """
    )
    return


@app.cell
def _():
    from typing import NamedTuple
    from nest.lib.hl_api_sonata import SonataNetwork
    from nest.lib.hl_api_nodes import Create as NestCreate
    from nest.lib.hl_api_connections import Connect as NestConnect
    from nest.lib.hl_api_types import NodeCollection

    class NestSonata(NamedTuple):
        net : SonataNetwork | None = None
        spike_rec: NodeCollection | None = None
        multi_meter: NodeCollection | None = None


    def load_nest_sonata(
        nest_config_file: str = "./v1/config_nest.json",
    ):
        # Instantiate SonataNetwork
        sonata_net = SonataNetwork(nest_config_file)

        # Create and connect nodes
        node_collections = sonata_net.BuildNetwork()
        print("Node Collections", node_collections.keys())

        # Connect spike recorder to a population
        spike_rec = NestCreate("spike_recorder")
        NestConnect(node_collections["v1"], spike_rec)

        # Attach Multimeter
        multi_meter = NestCreate(
            "multimeter",
            params={
                # "interval": 0.05,
                "record_from": [
                    "V_m",
                    "I",
                    "I_syn",
                    "threshold",
                    "threshold_spike",
                    "threshold_voltage",
                    "ASCurrents_sum",
                ],
            },
        )
        NestConnect(multi_meter, node_collections["v1"])

        # Simulate the network
        # sonata_net.Simulate()
        return NestSonata(sonata_net, spike_rec, multi_meter)

    # nest_net = load_nest_sonata()
    return (
        NamedTuple,
        NestConnect,
        NestCreate,
        NestSonata,
        NodeCollection,
        SonataNetwork,
        load_nest_sonata,
    )


@app.cell
def _():
    #
    # nest_net.net.Simulate()
    return


if __name__ == "__main__":
    app.run()
