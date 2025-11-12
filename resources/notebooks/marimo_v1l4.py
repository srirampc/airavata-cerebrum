import marimo

__generated_with = "0.12.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import logging
    import matplotlib.pyplot as plt
    import marimo as mo
    import nest
    ##
    import awitree
    import airavata_cerebrum.recipe as cbm_recipe
    import airavata_cerebrum.view as cbm_view
    import airavata_cerebrum.view.motree as cbm_motree
    import airavata_cerebrum.model.structure as cbm_structure
    import airavata_cerebrum.util as cbm_utils
    import mousev1.model as mousev1
    return (
        awitree,
        cbm_motree,
        cbm_recipe,
        cbm_structure,
        cbm_utils,
        cbm_view,
        json,
        logging,
        mo,
        mousev1,
        nest,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Setup the Model Recipe and the Recipe Templates

        Model recipe is defined by a recipe file named recipe.json and template file name recipe_template.json. 
        The files are placed in the recipe directory, canonically under the base directory of the model.

        ## recipe.json

        recipe.json contains two sections: (1) data2model_map and (2) source_data. 
        data2model_map maps between how the different parts of the model map to the source data. 
        The source_data describes how the source databases are connected and different operations such as filters are applied to the data available from the specific database.
        """
    )
    return


@app.cell
def _(cbm_motree, cbm_recipe, cbm_structure, mo):
    custom_mod_file = "./v1l4/recipe/custom_mod.json"
    m_base_dir = "./"
    m_name = "v1l4"
    rcp_files = {"recipe": ["recipe.json"], "templates": ["recipe_template.json"]}
    rcp_dir = "./v1l4/recipe/"
    main_rcp_file = "./v1l4/recipe/recipe.json"
    acquire_flag = False
    save_bmtk = False

    # recipe_dict = cbm_utils.io.load_json(main_rcp_file)
    # cmod_dict = cbm_utils.io.load_json(custom_mod_file)
    mdr_setup = cbm_recipe.init_recipe_setup(
        name=m_name,
        base_dir=m_base_dir,
        recipe_files=rcp_files,
        recipe_dir=rcp_dir,
    )

    tree_view_widths = [0.4, 0.6]
    cmod_struct = cbm_structure.Network.from_file(custom_mod_file)

    integ_explorer = cbm_motree.RecipeExplorer(mdr_setup, cmod_struct).build("V1L4")
    integ_tree, integ_panels = integ_explorer.view_components()
    integ_motree = mo.ui.anywidget(integ_tree)
    return (
        acquire_flag,
        cmod_struct,
        custom_mod_file,
        integ_explorer,
        integ_motree,
        integ_panels,
        integ_tree,
        m_base_dir,
        m_name,
        main_rcp_file,
        mdr_setup,
        rcp_dir,
        rcp_files,
        save_bmtk,
        tree_view_widths,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Visualize Recipe Section

        Recipe consists of three sections: (1) Data Source Section ; (2) Data2Model mapper section; and (3) Custom Modifications. An integrated viewer displays the three sections as below: 

        ## Source Data

        Data provider module includes
          - Methods for defining the querying from the database and filter the data based on specific criteria. 
          - Utilities to visualize the data provider configurations in easy-to-understand explorer view inside Jupyter notebook with the relevant parameters displayed in the side panel.

        Construction of Layer 4 of Mouse V1 is shown below with three different data providers: 
          - Allen Cell Types database,
          - Allen Brain Cell Atlas and
          - AI Synaptic Physiology Database. 

        ## Data2Model Map

        Definitions for data2model includes two parts:

        1. *Locations:* Cell types database and the MERFISH atlas data map to neuron models and the distribution of neuron types, respectively
        2. *Connections:* AI synaptic physiology data is mapped to the connection probabilities between the pairs of neuron classes.     

        ### View for Locations

        Locations are defined hiearchially with each section defining the data links. For V1 layer 4, we map five different neuron types to the specific models and the region fractions.

        Using the templates we can view the links to the neurons and the source data with Jupter notebook widgets. Running the next cell should produce a visualization in the image below:

        ### View for Connections

        Connections are defined as section for each pair to neuron sets, with each section defining the data links. For V1 layer 4, we map eight different neuron pairs to the AI Syn. Phys. Data outputs.

        Using the templates we can view the links to the neuron pairs and the source data with Jupter notebook widgets. Running the next cell should produce a visualization in the image below:

        ## Custom Modification

        Custom modifications if defined byt the "custom_mod.json". It includes:
        - Users can provide details required for network construction that are either not available in the linked databases or over-ride specific information.
        - Utilities to visualize these modifications in Jupyter Notebook.

        For the Layer 4 of Mouse V1, custom modifications include 
        1. Dimensions of each region in the model.
        2. Connection Parameters not available with the AI Syn. Phys. database.
        3. Details of the networks that are external to V1  LGN and the  background networks.
        """
    )
    return


@app.cell(hide_code=True)
def _(cbm_view, integ_motree, integ_panels, mo, tree_view_widths):
    integ_selected = cbm_view.BaseTree.panel_selector(integ_motree, integ_panels)
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
        ## Mappers for *Model Recipe* to *SONATA Network*
        To realize model recipe 
        1. *Region Mapper* : Maps the location to a region in the network
        2. *Neuron Mapper* : Map the neuron details in the location to a neuron class within the region
        3. *Connection Mapper*: Map the connection data to a connection object

        In addition a *Network Builder* class is also defined that translates the model description to SONATA network.
        """
    )
    return


@app.cell
def _(cbm_recipe, cmod_struct, mdr_setup, mousev1):
    model_recipe = cbm_recipe.ModelRecipe(
        recipe_setup=mdr_setup,
        region_mapper=mousev1.V1RegionMapper,
        neuron_mapper=mousev1.V1NeuronMapper,
        connection_mapper=mousev1.V1ConnectionMapper,
        network_builder=mousev1.V1BMTKNetworkBuilder,
        mod_structure=cmod_struct,
        save_flag=True,
        write_duck=True,
    )

    # logging.basicConfig(level=logging.INFO)
    return (model_recipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Data Acquistition

        Data acquisition consists of the following two steps:

        ## Run Data Download Workflow

        After the model description is defined and updated with custom user modifications, the download workflow proceeds as follows:

        1. Download the data
        2. Apply the filters and transormations
        3. Map data to the model components
        4. Apply user modifications
        5. Build the SONATA file

        ## Run Download Post Operations

        After data is dowloaded, the data obtained from different databases need to processed separately:

        1.  In case of the Allen Cell Type database, the download step can be restricted only to obtain the metadata related to cell types. After the meta data is downloaded, we need to acquire the models of interest (3 LIF Models). We use the GLIF API from allensdk to download these 3LIF model with a explained variance threshold.
        2.  For data from Allen Brain Cell Atlas, we filter the data specific to only the VISp Layer 4
        3.  For AI Syn Phys. data, we select only the neuron pairs of our interest.

        We accomplish the above two steps by running acquire_source_data() function.
        """
    )
    return


@app.cell
def _(acquire_flag, model_recipe):
    #
    # Commenting to avoid downloading from the database
    #
    if acquire_flag:
        db_source_data = model_recipe.acquire_source_data()
    return (db_source_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Explore downloaded data

        Output data are stored in the description directory in json format (db_connect_output.json), which can be examine with json library. Here are examples of Region and Neuron fractions of the Allen Brain Atlas, and connectivity matrix downloaded from AI Syn. Phys. dataset
        """
    )
    return


@app.cell
def _(json):
    #
    with open("./v1l4/recipe/db_connect_output.json") as dbf:
        db_out_data = json.load(dbf)
    return db_out_data, dbf


@app.cell
def _(db_out_data):
    import airavata_cerebrum.dataset.abc_mouse as abcm
    abcm.DFBuilder.build(db_out_data['airavata_cerebrum.dataset.abc_mouse'])
    return (abcm,)


@app.cell
def _(db_out_data):
    import airavata_cerebrum.dataset.ai_synphys as aisp
    aisp.DFBuilder.build(db_out_data['airavata_cerebrum.dataset.ai_synphys'])
    return (aisp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mapping Source data

        In this step, the data downloaded is mapped to the locations and the connection pairs as mentioned in "Data2Model Map" section above.
        """
    )
    return


@app.cell
def _(logging, model_recipe):
    logging.basicConfig(level=logging.INFO)
    #
    msrc = model_recipe.map_source_data()
    return (msrc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Apply User Modification

        As mentioned in the "User Modification" section above user updates for the model are loaded from `./v1l4/description/custom_mod.json`.

        For V1L4, the user modification include dimension, additional connection properties, and properties of external netowrks
        """
    )
    return


@app.cell
def _(model_recipe):
    mnet = model_recipe.build_net_struct()
    mnet = model_recipe.apply_mod()
    return (mnet,)


@app.cell
def _(model_recipe, mousev1):
    bmtk_net_builder = mousev1.V1BMTKNetworkBuilder(model_recipe.network_struct)
    bmtk_net = bmtk_net_builder.build()
    return bmtk_net, bmtk_net_builder


@app.cell
def _(bmtk_net_builder, mdr_setup, save_bmtk):
    if save_bmtk:
        bmtk_net_builder.net.save(str(mdr_setup.network_dir))
        bmtk_net_builder.bkg_net.save(str(mdr_setup.network_dir))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Running the SONATA Network in NEST

        1. Convert downloaded models to NEST confirmed models
        2. Load SONATA network in NEST
        3. Run Simulation
        """
    )
    return


@app.cell
def _(nest):
    config_file = "./v1l4/config_nest.json"
    # Instantiate SonataNetwork
    sonata_net = nest.SonataNetwork(config_file)

    # Create and connect nodes
    node_collections = sonata_net.BuildNetwork()
    print("Node Collections", node_collections.keys())
    return config_file, node_collections, sonata_net


@app.cell
def _(nest, node_collections, sonata_net):
    # Connect spike recorder to a population
    spike_rec = nest.Create("spike_recorder")
    nest.Connect(node_collections["v1l4"], spike_rec)

    # Attach Multimeter
    multi_meter = nest.Create(
        "multimeter",
        params={
            # "interval": 0.05,
            "record_from": ["V_m", "I", "I_syn", "threshold", "threshold_spike", "threshold_voltage", "ASCurrents_sum"],
        },
    )
    nest.Connect(multi_meter, node_collections["v1l4"])

    # Simulate the network
    sonata_net.Simulate()
    return multi_meter, spike_rec


@app.cell
def _(multi_meter, plt):
    #
    dmm = multi_meter.get()
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]
    #
    plt.figure(1)
    plt.plot(ts, Vms)
    return Vms, dmm, ts


@app.cell
def _(plt, spike_rec):
    spike_data = spike_rec.events
    spike_senders = spike_data["senders"]
    ts2 = spike_data["times"]
    plt.figure(2)
    plt.plot(ts2, spike_senders, ".")
    plt.show()
    return spike_data, spike_senders, ts2


if __name__ == "__main__":
    app.run()
