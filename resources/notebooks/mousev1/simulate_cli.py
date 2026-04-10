import argparse
import numpy as np
import typing as t
from bmtk.simulator.pointnet.pyfunction_cache import synaptic_weight
from bmtk.simulator import pointnet

from nest.lib.hl_api_info import SetStatus

@synaptic_weight
def weight_function_recurrent(edges, src_nodes: t.Any, trg_nodes: t.Any):
    return edges["syn_weight"].values


@synaptic_weight
def weight_function_lgn(edges, src_nodes: t.Any, trg_nodes: t.Any):
    return weight_function_recurrent(edges, src_nodes, trg_nodes)


@synaptic_weight
def weight_function_bkg(edges, src_nodes: t.Any, trg_nodes: t.Any):
    return weight_function_recurrent(edges, src_nodes, trg_nodes)


def get_v1_node_nums(sim):
    node_ids = sim.net._node_sets["v1"]._populations[0]._node_pop.node_ids
    return len(node_ids)


def set_random_potentials(sim):
    node_nums = get_v1_node_nums(sim)
    random_potentials = np.random.uniform(low=-75.0, high=-55.0, size=node_nums)
    SetStatus(range(1, node_nums + 1), "V_m", random_potentials)


def main(config_file: str, n_thread: int):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, graph, n_thread=n_thread)
    # sim = pointnet.PointSimulator.from_config(configure, graph)

    # if you want to initialize the network with random membrane potentials,
    # uncomment the following line
    # set_random_potentials(sim)
    sim.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the (GLIF) V1 with the SONATA network file."
    )
    parser.add_argument(
        "-i",
        "--input_file",
        required=True,
        help="SONATA network configuration.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        default=8,
        type=int,
        help="SONATA network configuration.",
    )
    rargs = parser.parse_args()
    main(rargs.input_file, rargs.threads)
 
