import ast
import logging
import typing as t

#
import numpy as np
import scipy
import scipy.stats
import pandas as pd

#
from typing_extensions import override
from pathlib import Path
from jsonpath import JSONPointer
from bmtk.builder import NetworkBuilder
from bmtk.builder.node_pool import NodePool
from airavata_cerebrum.operations import netops
from airavata_cerebrum.dataset import abc_mouse
from mousev1.operations import (
    compute_pair_type_parameters,
    connect_cells,
    syn_weight_by_experimental_distribution,
    generate_positions_grids,
    get_filter_spatial_size,
    get_filter_temporal_params,
    select_bkg_sources,
    select_lgn_sources_powerlaw,
    lgn_synaptic_weight_rule
)
from airavata_cerebrum.model import structure


def _log():
    return logging.getLogger(__name__)


class V1RegionMapper(structure.RegionMapper):
    def __init__(self, name: str, region_desc: dict[str, dict[str, t.Any]]):
        self.name : str = name
        self.region_desc : dict[str, dict[str, t.Any]]= region_desc
        self.property_map  : dict[str, t.Any] = self.region_desc["property_map"]
        self.src_data : dict[str, t.Any] = self.property_map["airavata_cerebrum.dataset.abc_mouse"][0]

    @override
    def neuron_names(self) -> list[str]:
        return self.property_map["neurons"]

    def inh_fraction(self):
        return float(self.src_data["inh_fraction"])

    def region_fraction(self):
        return float(self.src_data["region_fraction"])

    @override
    def map(
        self, region_neurons: dict[str, structure.Neuron]
    ) -> structure.Region | None:
        return structure.Region(
            name=self.name,
            inh_fraction=self.inh_fraction(),
            region_fraction=self.region_fraction(),
            neurons=region_neurons,
        )


structure.RegionMapper.register(V1RegionMapper)


class V1NeuronMapper(structure.NeuronMapper):
    abc_ptr : JSONPointer = JSONPointer(
        "/airavata_cerebrum.dataset.abc_mouse/0"
    )
    ct_ptr : JSONPointer = JSONPointer(
        "/airavata_cerebrum.dataset.abm_celltypes"
    )
    ev_ptr : JSONPointer = JSONPointer(
        "/glif/neuronal_models/0/neuronal_model_runs/0/explained_variance_ratio"
    )
    id_ptr : JSONPointer = JSONPointer("/glif/neuronal_models/0/neuronal_model_runs/0/id")

    def __init__(self, name: str, desc: dict[str, dict[str, t.Any]]):
        self.name : str = name
        self.desc : dict[str, dict[str, t.Any]] = desc
        self.ct_data: list[dict[str, t.Any]] | None = None
        self.abc_data: dict[str, t.Any] | None = None
        if self.abc_ptr.exists(desc):
            self.abc_data = self.abc_ptr.resolve(desc)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.abc_data = None
        if self.abc_data is None:
            _log().error("Can not find ABC Pointer in Model Description")
        if self.ct_ptr.exists(desc):
            self.ct_data = self.ct_ptr.resolve(desc)   # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.ct_data = None
        if self.ct_data is None:
            _log().error("Can not find ABM CT Pointer in Model Description")

    @override
    def map(self) -> structure.Neuron | None:
        ntype = self.desc["property_map"]["ei"]
        if not self.abc_data:
            return structure.Neuron(name=self.name, ei=ntype)
        nfrac = self.abc_data["fraction"]  # type:  ignore
        if not self.ct_data:
            return structure.Neuron(name=self.name, ei=ntype, fraction=float(nfrac))
        neuron_models = {}
        for mdesc in self.ct_data:
            spec_id = mdesc["ct"]["specimen__id"]
            m_type = "point_process"
            m_template = "nest:glif_psc"
            params_file = str(spec_id) + "_glif_lif_asc_config.json"
            m_id = self.id_ptr.resolve(mdesc)
            property_map = {
                "id": str(self.id_ptr.resolve(mdesc)),
                "explained_variance_ratio": self.ev_ptr.resolve(mdesc),
            }
            nmodel = structure.NeuronModel(
                name=str(spec_id),
                id=str(m_id),
                m_type=m_type,
                template=m_template,
                dynamics_params=params_file,
                property_map=property_map,
            )
            neuron_models[nmodel.name] = nmodel
        return structure.Neuron(
            name=self.name,
            ei=ntype,
            fraction=float(nfrac),
            neuron_models=neuron_models,
        )


structure.NeuronMapper.register(V1NeuronMapper)


@t.final
class V1ConnectionMapper(structure.ConnectionMapper):
    def __init__(self, name: str, desc: dict[str, t.Any]):
        self.name : str= name
        self.pre, self.post = ast.literal_eval(self.name)
        self.desc = desc

    @override
    def map(self) -> structure.Connection | None:
        if "airavata_cerebrum.dataset.ai_synphys" in self.desc:
            conn_prob = self.desc["airavata_cerebrum.dataset.ai_synphys"][0][
                "probability"
            ]
            return structure.Connection(
                name=self.name,
                pre=self.pre,
                post=self.post,
                property_map={"A_literature": [conn_prob]},
            )
        else:
            return structure.Connection(
                name=self.name,
                pre=self.pre,
                post=self.post,
            )


structure.ConnectionMapper.register(V1ConnectionMapper)


class ABCRegionMapper:
    def __init__(self, name: str, region_desc: dict[str, t.Any]):
        self.name : str = name
        self.region_desc : dict[str, t.Any] = region_desc

    def map(
        self, neuron_struct: dict[str, structure.Neuron]
    ) -> structure.Region:
        return structure.Region(
            name=self.name,
            inh_fraction=float(self.region_desc[abc_mouse.INHIBITORY_FRACTION_COLUMN]),
            region_fraction=float(
                self.region_desc[abc_mouse.FRACTION_WI_REGION_COLUMN]
            ),
            neurons=neuron_struct,
        )


class ABCNeuronMapper:
    def __init__(self, name: str, desc: dict[str, t.Any]):
        self.name : str = name
        self.desc : dict[str, t.Any] = desc
        self.ei_type : t.Literal['e', 'i'] = desc["ei"]

    def map(self) -> structure.Neuron | None:
        frac_col = abc_mouse.FRACTION_COLUMN_FMT.format(self.name)
        return structure.Neuron(
            ei=self.ei_type,
            fraction=float(self.desc[frac_col])
        )


class V1BMTKNetworkBuilder:
    def __init__(
        self,
        net_struct: structure.Network,
        **kwargs: t.Any,
    ):
        self.net_struct: structure.Network = net_struct
        self.fraction: float = 1.0
        self.flat: bool = False
        if "fraction" in kwargs:
            self.fraction = kwargs["fraction"]
        if "flat" in kwargs:
            self.fraction = kwargs["flat"]

        self.min_radius: float = 1.0  # to avoid diverging density near 0
        self.radius: float = self.net_struct.dims["radius"] * np.sqrt(self.fraction)
        self.radial_range: list[float] = [self.min_radius, self.radius]
        self.net: NetworkBuilder = NetworkBuilder(self.net_struct.name)
        lgn_struct = self.net_struct.ext_networks["lgn"]
        self.lgn_net : NetworkBuilder = NetworkBuilder(lgn_struct.name)
        bkg_struct = self.net_struct.ext_networks["bkg"]
        self.bkg_net : NetworkBuilder = NetworkBuilder(bkg_struct.name)

    def add_model_nodes(
        self,
        location: str,
        loc_dims: dict[str, t.Any],
        pop_neuron: structure.Neuron,
        neuron_model: structure.NeuronModel,
    ):
        pop_name = pop_neuron.name
        pop_size = pop_neuron.N
        # if "N" not in model:
        #     # Assumes a 'proportion' key with a value from 0.0 to 1.0, N will be a proportion of pop_size
        #     model["N"] = model["proportion"] * pop_size
        #     del model["proportion"]
        pN = pop_size
        if neuron_model.N == 0:
            if neuron_model.proportion > 0:
                pN = int(pop_size * neuron_model.proportion)
            else:
                pN = int(float(pop_size) / float(len(pop_neuron.neuron_models)))
        else:
            pN = neuron_model.N
        ei = pop_neuron.ei
        if self.fraction != 1.0:
            # Each model will use only a fraction of the of the number of cells for each model
            # NOTE: We are using a ceiling function so there is atleast 1 cell of each type - however for models
            #  with only a few initial cells they can be over-represented.
            pN = int(np.ceil(self.fraction * pN))

        if self.flat:
            pN = 100
        #
        # create a list of randomized cell positions for each cell type
        depth_range = -np.array(loc_dims["depth_range"], dtype=float)
        positions = netops.generate_random_cyl_pos(pN, depth_range, self.radial_range)
        #
        # properties used to build the cells for each cell-type
        nsyn_lognorm_shape = pop_neuron.dims["nsyn_lognorm_shape"]
        nsyn_lognorm_scale = pop_neuron.dims["nsyn_lognorm_scale"]
        target_sizes = np.array([])
        if nsyn_lognorm_shape > 0:
            target_sizes = netops.generate_target_sizes(
                pN, nsyn_lognorm_shape, nsyn_lognorm_scale
            )
        nsyn_size_mean = 0
        if nsyn_lognorm_shape > 0:
            nsyn_size_mean = int(
                scipy.stats.lognorm(
                    s=nsyn_lognorm_shape, loc=0, scale=nsyn_lognorm_scale
                ).stats(moments="m")
            )
        print(
            pop_name, neuron_model.name, int(neuron_model.name),
            len(target_sizes), nsyn_size_mean  # type: ignore
        )
        node_props = {
            "N": pN,
            "node_type_id": int(neuron_model.name),  # model["node_type_id"],
            "model_type": neuron_model.m_type,  # model["model_type"],  #  "model_type": "point_process",
            "model_template": neuron_model.template,  # model["model_template"],
            "model_id": neuron_model.id,
            "dynamics_params": neuron_model.dynamics_params,
            "ei": ei,
            "location": location,
            "pop_name": pop_name,
            "population": self.net_struct.name,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "tuning_angle": np.linspace(0.0, 360.0, pN, endpoint=False),
            "target_sizes": target_sizes,
            "nsyn_size_shape": nsyn_lognorm_shape,
            "nsyn_size_scale": nsyn_lognorm_scale,
            "nsyn_size_mean": nsyn_size_mean,
            # "size_connectivity_correction":
        }
        # TODO:: Update for biophysical model
        # if model["model_type"] == "biophysical":
        #     # for biophysically detailed cell-types add info about rotations and morphology
        #     node_props.update(
        #         {
        #             # for RTNeuron store explicity store the x-rotations
        #             #   (even though it should be 0 by default).
        #             "rotation_angle_xaxis": np.zeros(N),
        #             "rotation_angle_yaxis": 2 * np.pi * np.random.random(N),
        #             # for RTNeuron we need to store z-rotation in the h5 file.
        #             "rotation_angle_zaxis": np.full(
        #                 N, model["rotation_angle_zaxis"]
        #             ),
        #             "model_processing": model["model_processing"],
        #             "morphology": model["morphology"],
        #         }
        #     )

        self.net.add_nodes(**node_props)  # pyright: ignore[reportArgumentType]

    def add_nodes(
        self,
    ) -> None:
        for location, loc_region in self.net_struct.locations.items():
            # --- Neuron ---
            for _, pop_neuron in loc_region.neurons.items():
                pop_size = pop_neuron.N
                if pop_size == 0:
                    continue
                # --- Neuron Model ---
                for _, neuron_model in pop_neuron.neuron_models.items():
                    self.add_model_nodes(
                        location, loc_region.dims, pop_neuron, neuron_model
                    )

    def add_connection_edges(
        self,
        connex_item: structure.Connection,
        connex_md: structure.ConnectionModel,
    ) -> None:
        node_type_id = int(connex_md.target_model_id)
        src_type = connex_item.pre  # row["source_label"]
        trg_type = connex_item.post  # row["target_label"]
        src_trg_params = compute_pair_type_parameters(
            str(src_type),
            str(trg_type),
            connex_item.property_map,
        )

        prop_query = ["x", "z", "tuning_angle"]
        src_criteria = {"pop_name": repr(src_type)}
        self.net.nodes()  # this line is necessary to activate nodes... (I don't know why.)
        source_nodes = NodePool(self.net, **src_criteria)
        source_nodes_df = pd.DataFrame(
            [{q: s[q] for q in prop_query} for s in source_nodes]
        )
        md_pmap = connex_md.property_map
        cx_pmap = connex_item.property_map

        # TODO: check if these values should be used
        # weight_fnc, weight_sigma = find_direction_rule(src_type, trg_type)
        if src_trg_params["A_new"] > 0.0 and len(source_nodes) > 0:
            print(src_type, trg_type, node_type_id,
                  len(source_nodes), src_trg_params)
            # tentative fix for non-negative inhibitory connections
            if cx_pmap["pre_ei"] == "i":
                pspsign = -1
            else:
                pspsign = 1
            cm = self.net.add_edges(
                source=src_criteria,
                target={"node_type_id": node_type_id},
                iterator="all_to_one",
                connection_rule=connect_cells,  # pyright: ignore[reportArgumentType]
                connection_params={
                    "params": src_trg_params,
                    "source_nodes": source_nodes_df,
                    "core_radius": self.net_struct.dims["core_radius"],
                },
                dynamics_params=md_pmap["params_file"],
                delay=connex_md.delay,
                weight_function="weight_function_recurrent",
                PSP_correction=np.abs(md_pmap["PSP_scale_factor"]) * pspsign,
                PSP_lognorm_shape=md_pmap["lognorm_shape"],
                PSP_lognorm_scale=md_pmap["lognorm_scale"],
                model_template="static_synapse",
            )

            cm.add_properties(
                ["syn_weight", "n_syns_"],
                rule=syn_weight_by_experimental_distribution,
                rule_params={
                    "src_type": src_type,
                    "trg_type": trg_type,
                    "src_ei": cx_pmap["pre_ei"],
                    "trg_ei": cx_pmap["post_ei"],
                    "PSP_correction": np.abs(md_pmap["PSP_scale_factor"]) * pspsign,
                    "PSP_lognorm_shape": md_pmap["lognorm_shape"],
                    "PSP_lognorm_scale": md_pmap["lognorm_scale"],
                    "connection_params": src_trg_params,
                },
                dtypes=[float, np.int64],
            )

    def add_lgn_nodes(
        self, X_grids:int=15, Y_grids:int=10,
        x_block:float=8.0, y_block:float=8.0
    ):
        X_len = x_block * X_grids  # default is 120 degrees
        Y_len = y_block * Y_grids  # default is 80 degrees
        xcoords = []
        ycoords = []
        lgn_region = self.net_struct.ext_networks["lgn"].locations["lgn"]
        for lgn_name, lgn_neuron in lgn_region.neurons.items():
            lgn_model = lgn_neuron.neuron_models[lgn_name]
            # Get position of lgn cells and keep track of the averaged location
            # For now, use randomly generated values
            total_N = lgn_neuron.N * X_grids * Y_grids
            # Get positional coordinates of cells
            positions = generate_positions_grids(
                lgn_neuron.N, X_grids, Y_grids, X_len, Y_len
            )
            xcoords += [p[0] for p in positions]
            ycoords += [p[1] for p in positions]
            # Get spatial filter size of cells
            filter_sizes = get_filter_spatial_size(
                lgn_neuron.N, X_grids, Y_grids,
                lgn_neuron.dims["size_range"]
            )
            # Get filter temporal parameters
            filter_params = get_filter_temporal_params(
                lgn_neuron.N, X_grids, Y_grids, lgn_name
            )
            # Get tuning angle for LGN cells
            # tuning_angles = get_tuning_angles(params['N'], X_grids, Y_grids, model)
            self.lgn_net.add_nodes(
                N=total_N,
                pop_name=lgn_model.name,
                model_type="virtual",
                ei="e",
                location="LGN",
                x=positions[:, 0],
                y=positions[:, 1],
                spatial_size=filter_sizes,
                kpeaks_dom_0=filter_params[:, 0],
                kpeaks_dom_1=filter_params[:, 1],
                weight_dom_0=filter_params[:, 2],
                weight_dom_1=filter_params[:, 3],
                delay_dom_0=filter_params[:, 4],
                delay_dom_1=filter_params[:, 5],
                kpeaks_non_dom_0=filter_params[:, 6],
                kpeaks_non_dom_1=filter_params[:, 7],
                weight_non_dom_0=filter_params[:, 8],
                weight_non_dom_1=filter_params[:, 9],
                delay_non_dom_0=filter_params[:, 10],
                delay_non_dom_1=filter_params[:, 11],
                tuning_angle=filter_params[:, 12],
                sf_sep=filter_params[:, 13],
            )
        return self.lgn_net

    def add_lgn_v1_edges(
        self, x_len:float=240.0, y_len:float=120.0
    ):
        # skipping the 'locations' (e.g. VisL1) key and make a population-based
        # (e.g. i1Htr3a) dictionary
        # in this file, the values are specified for each target model
        lgn_mean = (x_len / 2.0, y_len / 2.0)
        prop_query = ["node_id", "x", "y", "pop_name", "tuning_angle"]
        lgn_nodes = pd.DataFrame(
            [{q: s[q] for q in prop_query} for s in self.lgn_net.nodes()]
        )
        # this regular expression is picking up a number after TF
        lgn_nodes["temporal_freq"] = lgn_nodes["pop_name"].str.extract(r"TF(\d+)")
        # make a complex version beforehand for easy shift/rotation
        lgn_nodes["xy_complex"] = lgn_nodes["x"] + 1j * lgn_nodes["y"]
        for lgn_conn_name, lgn_connect in self.net_struct.connections.items():
            _, v1_neuron_name = ast.literal_eval(lgn_conn_name)
            v1_neuron: structure.Neuron | None = self.net_struct.find_neuron(v1_neuron_name)
            if v1_neuron is None:
                print(f"LGN :: Unknown neuron name: {v1_neuron_name}")
                continue
            for _, lgn_conn_model in lgn_connect.connect_models.items():
                # target_pop_name = row["population"]
                target_model_id = int(lgn_conn_model.target_model_id)
                e_or_i = v1_neuron.ei
                if e_or_i == "e":
                    sigma = [0.0, 150.0]
                elif e_or_i == "i":
                    sigma = [0.0, 1e20]
                else:
                    # Additional care for LIF will be necessary if applied for Biophysical
                    raise BaseException(
                        f"Unknown e_or_i value: {e_or_i} from {v1_neuron.name}"
                    )
                # LGN is configured based on e4 response. Here we use the mean target sizes of
                # the e4 neurons and normalize all the cells using these values. By doing this,
                # we can avoid injecting too much current to the populations with large target
                # sizes.
                # Hard-coding e4other's properties
                lognorm_shape = 0.345
                lognorm_scale = 2188.765
                e4_mean_size = np.exp(np.log(lognorm_scale) + (lognorm_shape**2) / 2)
                edge_params = {
                    "source": self.lgn_net.nodes(),
                    "target": self.net.nodes(node_type_id=target_model_id),
                    "iterator": "all_to_one",
                    "connection_rule": select_lgn_sources_powerlaw,
                    "connection_params": {"lgn_mean": lgn_mean, "lgn_nodes": lgn_nodes},
                    "dynamics_params": lgn_conn_model.dynamics_params,
                    "delay": 1.7,
                    "weight_function": "weight_function_lgn",
                    "weight_sigma": sigma,
                    "model_template": "static_synapse",
                }
                cm = self.lgn_net.add_edges(**edge_params)  # pyright: ignore[reportArgumentType]
                cm.add_properties(
                    "syn_weight",
                    rule=lgn_synaptic_weight_rule,
                    rule_params={
                        "base_weight": lgn_conn_model.property_map["syn_weight_psp"],
                        "mean_size": e4_mean_size,
                    },
                    dtypes=float,
                )
        return self.lgn_net

    def add_bkg_nodes(self):
        bkg_struct = self.net_struct.ext_networks["bkg"]
        bkg_region = bkg_struct.locations["bkg"]
        bkg_neuron = bkg_region.neurons["bkg"]
        n_bkg = bkg_struct.ncells
        self.bkg_net.add_nodes(
            # N=1,
            N=n_bkg,
            pop_name=bkg_neuron.name,
            ei=bkg_neuron.ei,
            location=bkg_region.name,
            model_type=bkg_neuron.neuron_models["bkg"].m_type,
            x=np.zeros(n_bkg),
            y=np.zeros(n_bkg),  # are these necessary?
            # x=[-91.23767151810344],
            # y=[233.43548226294524],
        )
        return self.bkg_net

    def add_bkg_edges(self):
        bkg_struct: structure.ExtNetwork = self.net_struct.ext_networks["bkg"]
        # this file should contain the following parameters:
        # model_id (of targets), syn_weight_psp, dynamics_params, nsyns
        bkg_net_nodes = self.bkg_net.nodes()
        for _, bkg_connect in bkg_struct.connections.items():
            for _, bkg_model in bkg_connect.connect_models.items():
                nmodel_id = int(bkg_model.target_model_id)
                target_nodes = self.net.nodes(node_type_id=nmodel_id)
                if len(target_nodes) == 0:
                    continue
                print("bkg", bkg_connect.name, nmodel_id,
                      bkg_model.property_map,
                      len(bkg_net_nodes),
                      len(target_nodes))
                edge_params = {
                    "source": bkg_net_nodes,
                    "target": target_nodes,
                    # "connection_rule": lambda s, t, n: n,
                    "connection_rule": select_bkg_sources,
                    "iterator": "all_to_one",
                    "connection_params": {
                        "n_syns": bkg_model.property_map["nsyns"],
                        "n_conn": bkg_connect.property_map["n_conn"],
                    },
                    "dynamics_params": bkg_model.dynamics_params,  # row["dynamics_params"],
                    # "syn_weight": row["syn_weight_psp"],
                    "syn_weight": bkg_model.property_map[
                        "syn_weight"
                    ],  # row["syn_weight"],
                    "delay": 1.0,
                    "model_template": "static_synapse",
                    # "weight_function": "ConstantMultiplier_BKG",
                    "weight_function": "weight_function_bkg",
                }
                self.bkg_net.add_edges(**edge_params) # pyright: ignore[reportArgumentType]
        return self.bkg_net

    def add_edges(
        self,
    ):
        for _, connex_item in self.net_struct.connections.items():
            for _, connex_md in connex_item.connect_models.items():
                # Build model from connection information
                self.add_connection_edges(connex_item, connex_md)

    def build(
        self,
    ) -> NetworkBuilder:
        self.add_nodes()
        self.add_edges()
        # self.add_lgn_nodes()
        # self.add_lgn_v1_edges()
        self.add_bkg_nodes()
        self.add_bkg_edges()
        return self.net

    def save(self, network_dir: str | Path):
        self.net.save(str(network_dir))
        self.bkg_net.save(str(network_dir))

