# README

## Cerebrum Tools

Airavata Cerebrum is a suite of tools designed to effectively build data-driven
computational models. Key components of the software suite are as follows:

- **Data Providers :** Tools to query, filter, and aggregate data from online
  databases and brain atlases.
- **Data2Model Mappers :** Tools to map acquired data to the different components
  of the model through a recipe defined in a plain-text JSON/YAML format.
- **Model Recipe Editors :** Tools to add, remove and edit models components that can
  transform the model as per the needs/discoveries/innovations of the user.
- **Recipe Serializers :** Tools to build and save models in the SONATA format
  that enables the community to reproduce, reuse and remix models
  previously with Cerebrum.

### Abstract Model Definition

The main use case of Airavata Cerebrum is to build a computational model of
the brain or a specific region of interest within the brain, with the overall
aim to study the structure and the computations,
under realistic biological constraints.

Such a computational model requires the following building blocks:

1. _Neurons - Cell classes, and cell models_: Neurons are the indivisible
   units of such models and at the fundamental level, the model needs to define:
   (a) Cell Classes of neuron and (b) Cells models of neuron.

2. _Connectivity Models :_ Inter-region and Intra-region connectivity model based
   on the connectivity distributions.

3. _Distribution Models :_ Description of (a) the proportions of the cell classes
   within each region of interest, and (b) Distribution of the synapses.

A data-driven model builder uses building blocks defined based on the wealth of
data extracted from the extensive and detailed maps of the brain made
publicly available in large online databases (e.g. Brain Atlas).

Airavata Cerebrum enables a structured approach to author computational models
of brain and other neuron systems that simplifes the complete model
development cycle starting from data acquistion to model evaluation.

## Airavata Cerebrum

Builing a computational model with Cerebrum requires:

1. A Model Recipe: A _Cerebrum Recipe_ is collection of JSON or YAML files
   that is used to define a inital model skeleton. The files declare
   - Data sources,
   - Data acquistion parameters,
   - Data filtering/aggregating workflow,
   - Data2Model mapping rules and their parameters, and
   - Custom modifications that add/revise the purely data-driven model.

2. Model Builder classes: _python_ classes that implement the follwing functions:
   (i) Mapping functions that use the filtered data and the initial skeleton
   to generate the finalized model skeleton, and (ii) Builder functions
   that generates every neuron and the connection based on the model skeleton,
   to serialize as SONATA network file.

In the following sections, we describe the recipe and the model builder with
the Allen Institute V1 L4 model as an example.

### Recipe Description

At the top level, a recipe has two sections (a) Data source section and
(b) Data2Model mapping section. Additionally, a recipe can also include a
custom modifications file, that describes the desired updates and additions to the
model.

#### Recipe Data Sources

Data source section of the recipe
The following snippet shows a top-level view of the data sources section for
Allen Institute V1 L4 model.

```YAML
source_data:
  airavata_cerebrum.dataset.abm_celltypes:
    - ...
  airavata_cerebrum.dataset.abc_mouse:
    - ...
  airavata_cerebrum.dataset.ai_synphys:
    - ...
```

The V1 L4 model acquires data from three different sources: (a) Allen Institute
Cell Types database, (b) Allen Mouse Brain Atlas, and (c) Allen Insitute Synaptic
Physiology Database. For each data source, the recipe should describe the
workflow for acquiring the data.

Each data source stub describes the database from which the data is accessed and
a sequence of workflow steps that processes the data.
As an example data source recipe, show below is the complete description for
acquiring data from the _Allen Syn Phys Database_.

```YAML
source_data:
  airavata_cerebrum.dataset.abm_celltypes:
    - ...
  airavata_cerebrum.dataset.abc_mouse:
    - ...
  airavata_cerebrum.dataset.ai_synphys:
    label: AI Syn Phys
    db_connect:
      label: Query Allen Syn Phys Database
      workflow:
      - name: airavata_cerebrum.dataset.ai_synphys.AISynPhysQuery
        label: AI Syn. Phys. Query
        type: query
        init_params:
          download_base: ./cache/ai_synphys/
        exec_params:
          layer:
          - L4
    post_ops:
      workflow: []
```

The recipe's workflow, shown above, declares that the **airavata_cerebrum.dataset.ai_synphys.AISynPhysQuery**
query class is used to uery the Allen Syn. Phys. database with the initialization
paramters **download_base: ./cache/ai_synphys/** and the query execution paramters
**layer: [L4]**. Execution of the query downloads the connection probabilities
corresponding to L4 of the primay visual cortex. A table shown below is constructed
from the downloaded data.

| pre_synapse | post_synapse | connect_prob |
| ----------- | ------------ | ------------ |
| "L4-Pyr"    | "L4-Pyr"     | 0.125985     |
| "L4-Pyr"    | "L4-Pvalb"   | 0.198113     |
| "L4-Pyr"    | "L4-Sst"     | 0.045855     |
| "L4-Pyr"    | "L4-Vip"     | 0.049624     |
| "L4-Pvalb"  | "L4-Pyr"     | 0.356604     |
| …           | …            | …            |
| "L4-Sst"    | "L4-Vip"     | 0.306504     |
| "L4-Vip"    | "L4-Pyr"     | 0.0          |
| "L4-Vip"    | "L4-Pvalb"   | 0.025412     |
| "L4-Vip"    | "L4-Sst"     | 0.18856      |
| "L4-Vip"    | "L4-Vip"     | 0.03499      |

Currently, Cerebrum supports only three databases that is used in the mouse V1 models.
However, the recipe declaration can accept custom query classes written by user
(in the name field) as long as it outputs the data in JSON format.
Custom query classes should be a subclass of **airavata_cerebrum.DbQuery\[QInitParams, QExecParams\]**
and implement the run method which returns an iterator of results.
Instances of **QInitParams** and **QExecParams** are provided as the input
arguments for the constructor method and the run method respectively.
The query classes in the **airavata_cerebrum.dataset** module can be used as examples.

Cerebrum also has multiple different filters implemented that simplifies data selection
within the downloaded data. These classes can also be added as steps in the
workflow sub-section of the recipe. Users can also provide their own custom
classes as that implement any kind of transformation.
Custom data transform classes should be a subclass of
**airavata_cerebrum.OpXFormer\[XInitParams, XExecParams\]** and implement the
xform method which accepts, in addition to the execution parameters,
an input one or more iterators , and returns an iterator of processed results.
Instances of **XInitParams** and **XExecParams** are provided as the
arguments for the constructor method and the xform method respectively.
The transform classes in the **airavata_cerebrum.operations** module can be used as examples.

Data collected and processed from the databases are
also stored as both a JSON text file and, optionally, can be saved as a **duckdb** database.

#### Recipe Data2Model maps

At the top level the **data2model_map** section has two sub-sections: locations
and connections. **locations** sub-section describe the data mapping for each location,
region and the neuron types for the model. **connections** describe the mapping
for each of the connections.
The mapping for each location and connection describes how the data collected
from the data acquired from different sources should be mapped to the
basic structure of the model.

The following snippet shows a high-level overview of the data2model_map section
of V1 L4 model.

```YAML
data2model_map:
  locations:
    VISp4:
      ('4', 'Pvalb'):
         - ...
      ('4', 'Rorb'):
         - ...
      ('4', 'Scnn1a'):
         - ...
      ('4', 'Sst'):
         - ...
      ('4', 'Vip'):
         - ...
  connections:
    (('4', 'Pvalb'), ('4', 'Pvalb')):
      - ...
    (('4', 'Pvalb'), ('4', 'Sst')):
      - ...
    (('4', 'Pvalb'), ('4', 'Vip')):
      - ...
    (('4', 'Sst'), ('4', 'Pvalb')):
      - ...
    (('4', 'Sst'), ('4', 'Sst')):
      - ...
    (('4', 'Sst'), ('4', 'Vip')):
      - ...
    (('4', 'Vip'), ('4', 'Pvalb')):
      - ...
```

The following snippet shows part of the mapping subsection for the V1 L4 model.
The snippet shows the recipe that (i) the extract the
proportions of the neurons form the Brain Atlas MERFISH data; and (ii) the
cell type models are selected from Allen Cell Types DB for the cell class
included in V1 L4.

```YAML
  locations:
    VISp4:
      ('4', 'Pvalb'):
        property_map:
          ei: i
        source_data:
          airavata_cerebrum.dataset.abc_mouse:
            label: Brain Atlas:Cell Fraction
            workflow:
            - name: airavata_cerebrum.operations.abc_mouse.ABCDbMERFISH_CCFFractionFilter
              type: xform
              label: Select Region/Type
              exec_params:
                cell_type: Pvalb
                region: VISp4
              init_params: {}
          airavata_cerebrum.dataset.abm_celltypes:
            label: Cell Types DB:Filter
            workflow:
            - name: airavata_cerebrum.operations.abm_celltypes.CTPropertyFilter
              type: xform
              label: Select Line
              exec_params:
                key: ct
                line: Pvalb-IRES
                reporter_status: positive
              init_params: {}
      ('4', 'Vip'):
        - ...
```

The following snippet shows part of the connection mapping subsection for the
V1 L4 model.
The snippet shows the recipe that the connection probabilities for the
specific connection is acquired from the Allen Synaptic Phys. Database.

```YAML
  connections:
    (('4', 'Pvalb'), ('4', 'Pvalb')):
      source_data:
        airavata_cerebrum.dataset.ai_synphys:
          label: AI Syn Phys
          workflow:
          - name: airavata_cerebrum.operations.ai_synphys.AISynPhysPairFilter
            label: Filter by Pair
            type: xform
            exec_params:
              post: L4-Pvalb
              pre: L4-Pvalb
            init_params: {}
```

#### Recipe Custom Modifications

Custom modifications has three sections: locations, connections, and
ext_networks. locations and connections describe the details about the
neurons and the synapse connections. ext_networks provides the details about
the external networks that recipe input stimulus and how they connect to the model.

A snippet of the locations and connections for the V1 L4 model is shown below.
The locations sub-section includes the dimensions and the connections include the
paramters for the connection models.

```YAML
locations:
  VISp4:
    name: VISp4
    dims:
      depth_range:
      - 357
      - 505
    neurons:
      ...
connections:
  (('4', 'Pvalb'), ('4', 'Pvalb')):
    connect_models:
      (('4', 'Pvalb', ''), ('4', 'Pvalb', '478793814')):
        delay: 1.01
        name: (('4', 'Pvalb', ''), ('4', 'Pvalb', '478793814'))
        property_map:
          PSP_scale_factor: 0.037
          lognorm_scale: 0.5199354476
          lognorm_shape: 0.6828453321
          params_file: pv_to_pv.json
          synphys_mean: 0.6521111955
        source_model_id: ''
        target_model_id: '478793814'
        weight_max: -6.3513513514
      (('4', 'Pvalb', ''), ('4', 'Pvalb', '478958894')):
        ...
```

A snippet of ext_networks for the V1 L4 model is shown below. It describes the
details of background and LGN layer which are connected to the model and
recieve the input stimulus.

```YAML
ext_networks:
  bkg:
    connections:
      - ...
    locations:
      bkg:
        name: SG_001
        inh_fraction: 0.0
        ncells: 100
         - ...
  lgn:
    connections:
      - ...
    locations:
      lgn:
        name: lgn
        inh_fraction: 0.0
         - ...
```

### Model Builders Description

After processing the recipe for data acquistion, mapping and the custom
modifications, an initial skeleton is generated. The inital model skeleton
is tree represented with a _python_ dict
Building the final network from the initial skeleton proceeds in two phases.
In the first phase, a final model skeleton is built.
The final model skeletons are represented in _python_ as instances of
data model classes derived from the _pydantic_ library.
For building the final skeletion, requires three class

1. Neuron mapping class that is a a subclass of structure.NeuronMapper.
   The class has the mapping functions to accept the neuron data
   from the initial skeleton and generates a data object representing a Neuron,
   with a **structure.Neuron** object.
2. Region mapping class that is a subclass of structure.RegionMapper.
   The class accepts a the neuron data and **structure.Neuron** objects, and
   generats a data object representing a Region, with a **structure.Region** object.
3. Connection mapping class that a subclass of structure.ConnectionMapper.
   The class has the mapping functions to accept the connection data
   from the initial skeleton and generates a data object representing a Connection,
   with a **structure.Connection** object.

For the final step, a builder method that accepts the final model and generates
all location of the all the neurons, synapses between the neurons and saves them
in the SONATA format.
