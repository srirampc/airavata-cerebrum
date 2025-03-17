import argparse
import json
import os
import pathlib
import typing
import numpy as np
import bmtk.simulator.pointnet.glif_utils as glif_utils

def main(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cfg_fnames = os.listdir(input_dir)
    for fxname in cfg_fnames:
        in_fxname = pathlib.Path(input_dir, fxname)
        with open(in_fxname) as ifx:
            cfg_db = json.load(ifx)
        cvtx = glif_utils.converter_builtin("nest:glif_lif_asc_psc", cfg_db)
        if not cvtx:
            continue
        n_dict : dict[str, typing.Any] = cvtx[1]
        #del n_dict["tau_syn"]
        for kx, valx in n_dict.items():
            if valx.__class__ == np.ndarray:
                n_dict[kx] = list(valx)
        out_fname = pathlib.Path(output_dir, fxname)
        with open(out_fname, "w") as ofx:
            json.dump(n_dict, ofx, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Cell Models to format suitable BMTK"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output Directory for the cell models.",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory of the point network models.",
    )
    args = parser.parse_args()
    # input_dir = "./v1l4/components/point_neuron_models/"
    # output_dir = "./v1l4/components/cell_models/"
    print(args)
    main(args.input_dir, args.output_dir)
