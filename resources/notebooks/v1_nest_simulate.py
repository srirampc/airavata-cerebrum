import argparse
#
from typing import NamedTuple
#
from nest.lib.hl_api_sonata import SonataNetwork
from nest.lib.hl_api_nodes import Create as NestCreate
from nest.lib.hl_api_connections import Connect as NestConnect
from nest.lib.hl_api_types import NodeCollection

#@synaptic_weight
# def weight_function_recurrent(edges, src_nodes, trg_nodes):
#     return edges["syn_weight"].values


#@synaptic_weight
# def weight_function_bkg(edges, src_nodes, trg_nodes):
#     return weight_function_recurrent(edges, src_nodes, trg_nodes)


class NestSonata(NamedTuple):
    net : SonataNetwork | None = None
    spike_rec: NodeCollection | None = None
    multi_meter: NodeCollection | None = None


def load_nest_sonata(
    nest_config_file: str,
    collections_name: str
):
    # Instantiate SonataNetwork
    sonata_net = SonataNetwork(nest_config_file)

    # Create and connect nodes
    node_collections = sonata_net.BuildNetwork()
    print("Node Collections", node_collections.keys())

    # Connect spike recorder to a population
    spike_rec = NestCreate("spike_recorder")
    NestConnect(node_collections[collections_name], spike_rec)

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
    NestConnect(multi_meter, node_collections[collections_name])

    # Simulate the network
    # sonata_net.Simulate()
    return NestSonata(sonata_net, spike_rec, multi_meter)


def main(config_file: str, collections_name: str,
         output_dir: str, n_thread: int):
    nest_net = load_nest_sonata(config_file, collections_name)
    # Simulate the network
    nest_net.net.Simulate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the pointnet simulation with the given config file."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./v1/output/",
        help="This option will override the output directory specified in the config file.",
    )
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default="./v1/config_nest.json",
        help="The config file to use for the simulation.",
    )
    parser.add_argument(
        "collections_name",
        type=str,
        nargs="?",
        default="v1",
        help="Node collection name.",
    )
    parser.add_argument(
        "-n", "--n_thread", type=int, default=1, help="Number of threads to use."
    )
    args = parser.parse_args()

    main(
        args.config_file,
        args.collections_name,
        args.output_dir,
        args.n_thread,
    )
