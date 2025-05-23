#
# Code copied as it is from V1 Model
#
#
import os
import json
import math
import pathlib
import typing as t
import numpy as np
import numpy.typing as npt
import h5py
import pandas as pd
#
from math import sqrt, exp, log
from scipy.stats import yulesimon, multivariate_normal
from scipy.special import erfinv
# from numba import njit, jit
#
import bmtk.simulator.pointnet.glif_utils as glif_utils

LGN_SHIFT = {
    "sON_TF1": -1.0,
    "sON_TF2": -1.0,
    "sON_TF4": -1.0,
    "sON_TF8": -1.0,
    "sOFF_TF1": -1.0,
    "sOFF_TF2": -1.0,
    "sOFF_TF4": -1.0,
    "sOFF_TF8": -1.0,
    "sOFF_TF15": -1.0,
    "tOFF_TF4": 1.0,
    "tOFF_TF8": 1.0,
    "tOFF_TF15": 1.0,
    "sONsOFF_001": 0,
    "sONtOFF_001": 0,
}

LGN_PARAMS_POPS = [
    "('1', 'Htr3a')", 
    "('2/3', 'Cux2')", "('2/3', 'Pvalb')", "('2/3', 'Sst')", "('2/3', 'Vip')",
    "('4', 'Nr5a1')", "('4', 'Rorb')", "('4', 'Scnn1a')",
    "('4', 'other')", "('4', 'Pvalb')", "('4', 'Sst')", "('4', 'Vip')",
    "('5', 'IT')", "('5', 'ET')", "('5', 'NP')", "('5', 'Pvalb')",
    "('5', 'Sst')", "('5', 'Vip')",
    "('6', 'Ntsr1')", "('6', 'Pvalb')", "('6', 'Sst')", "('6', 'Vip')"
]
LGN_PARAMS_COLUMNS =[
    'pop_name', 'poisson_parameter', 'sON_ratio',
    'synapse_ratio_against_e4', 'yule_parameter'
] 
LGN_PARAMS_ARRAY = np.array(
    [["('1', 'Htr3a')", 2.0, 0.75, 0.098, 3.7],
     ["('2/3', 'Cux2')", 1.5, 0.9, 0.63, 4.3],
     ["('2/3', 'Pvalb')", 2.0, 0.75, 0.87, 2.4],
     ["('2/3', 'Sst')", 2.0, 0.75, 0.24, 3.7],
     ["('2/3', 'Vip')", 2.0, 0.75, 0.42, 4.7],
     ["('4', 'Nr5a1')", 2.0, 0.9, 1.0, 3.5],
     ["('4', 'Rorb')", 2.0, 0.9, 1.0, 3.5],
     ["('4', 'Scnn1a')", 2.0, 0.9, 1.0, 3.5],
     ["('4', 'other')", 2.0, 0.9, 1.0, 3.5],
     ["('4', 'Pvalb')", 2.0, 0.75, 2.76, 2.4],
     ["('4', 'Sst')", 2.0, 0.75, 0.92, 3.7],
     ["('4', 'Vip')", 2.0, 0.75, 0.79, 4.7],
     ["('5', 'IT')", 1.5, 0.5, 0.35, 5.6],
     ["('5', 'ET')", 1.5, 0.5, 0.5, 4.6],
     ["('5', 'NP')", 1.5, 0.5, 0.053, 7.6],
     ["('5', 'Pvalb')", 2.0, 0.5, 1.49, 2.4],
     ["('5', 'Sst')", 2.0, 0.5, 0.46, 3.7],
     ["('5', 'Vip')", 2.0, 0.5, 0.27, 4.7],
     ["('6', 'Ntsr1')", 1.5, 0.9, 0.11, 4.3],
     ["('6', 'Pvalb')", 2.0, 0.75, 0.22, 2.4],
     ["('6', 'Sst')", 2.0, 0.75, 0.072, 3.7],
     ["('6', 'Vip')", 2.0, 0.75, 0.018, 4.7]],
    dtype=object)
LGN_PARAMS = pd.DataFrame(
    data=LGN_PARAMS_ARRAY, columns=LGN_PARAMS_COLUMNS, index=LGN_PARAMS_POPS
)

LGN_BEST_FIT = {
    "sON_TF1": {
        'sopt_msg': ('After ', '34', 'function evaluations, TNC returned:',
                             'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 34,
        'sopt_res': np.array([  1.00003093,  -0.44616057,  56.52557576, 137.44093312]),
        'init_prms': [3.5, -2.0, 50.0, 140.0],
        'opt_wts': np.array([ 1.00003093, -0.44616057]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([ 0, 15]),
        'opt_kpeaks': np.array([ 56.52557576, 137.44093312])
    }, # best chosen fit for sON 1 Hz
    "sON_TF2" : {
        'sopt_msg': ('After ', '25', 'function evaluations, TNC returned:',
          'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 25,
        'sopt_res': np.array([  4.28233151,  -1.42754511,  31.20843844, 137.06239706]),
        'init_prms': [3.5, -2.0, 50.0, 140.0],
        'opt_wts': np.array([ 4.28233151, -1.42754511]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([0, 5]),
        'opt_kpeaks': np.array([ 31.20843844, 137.06239706])
    },# best chosen fit for sON 2 Hz
    "sON_TF4" : {
        'sopt_msg': ('After ', '13', 'function evaluations, TNC returned:',
                     'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 13,
        'sopt_res': np.array([ 3.5940406 , -1.81458312, 30.75886854, 58.35845979]),
        'init_prms': [3.5, -2.0, 30.0, 60.0],
        'opt_wts': np.array([ 3.5940406 , -1.81458312]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([ 0, 25]),
        'opt_kpeaks': np.array([30.75886854, 58.35845979])
    }, # best chosen fit for sON 4 Hz
    "sON_TF8" : {
        'sopt_msg': ('After ', '20', 'function evaluations, TNC returned:',
         'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 20,
        'sopt_res': np.array([ 3.70525968, -1.75383971, 12.94103321, 28.27807659]),
        'init_prms': [3.5, -2.0, 10.0, 20.0],
        'opt_wts': np.array([ 3.70525968, -1.75383971]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([ 0, 25]),
        'opt_kpeaks': np.array([12.94103321, 28.27807659])
    }, # best chosen fit for sON 8 Hz
    "sOFF_TF1" : {
        'sopt_msg': ('After ', '33', 'function evaluations, TNC returned:',
         'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 33,
        'sopt_res': np.array([  2.11424029,  -1.22152557,  54.48070428, 127.92115512]),
        'init_prms': [3.5, -2.0, 50.0, 140.0],
        'opt_wts': np.array([ 2.11424029, -1.22152557]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([0, 5]),
        'opt_kpeaks': np.array([ 54.48070428, 127.92115512])
    }, # best chosen fit for sOFF 1 Hz
    "sOFF_TF2" : {
        'sopt_msg': ('After ', '23', 'function evaluations, TNC returned:',
         'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 23,
        'sopt_res': np.array([ 3.18257937, -1.14071203, 19.84561655, 92.02360682]),
        'init_prms': [3.5, -2.0, 10.0, 100.0],
        'opt_wts': np.array([ 3.18257937, -1.14071203]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([ 0, 25]),
        'opt_kpeaks': np.array([19.84561655, 92.02360682])
    }, # best chosen fit for sOFF 2 Hz
    "sOFF_TF4" : {
        'sopt_msg': ('After ', '36', 'function evaluations, TNC returned:',
                     'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 36,
        'sopt_res': np.array([ 3.22981488, -1.65879752, 22.24619766, 56.10036268]),
        'init_prms': [3.5, -2.0, 10.0, 60.0],
        'opt_wts': np.array([ 3.22981488, -1.65879752]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([ 0, 15]),
        'opt_kpeaks': np.array([22.24619766, 56.10036268])
    }, # best chosen fit for sOFF 4 Hz
    "sOFF_TF8" : {
        'sopt_msg': ('After ', '27', 'function evaluations, TNC returned:',
         'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 27,
        'sopt_res': np.array([ 3.2506479 , -1.68380989, 18.03400572, 51.21537413]),
        'init_prms': [3.5, -2.0, 10.0, 60.0],
        'opt_wts': np.array([ 3.2506479 , -1.68380989]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([ 0, 15]),
        'opt_kpeaks': np.array([18.03400572, 51.21537413])
    }, # best chosen fit for sOFF 8 Hz
    "sOFF_TF15" : {
        'sopt_msg': ('After ', '28', 'function evaluations, TNC returned:',
         'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 28,
        'sopt_res': np.array([10.        , -3.26057123,  4.73623985, 21.59707569]),
        'init_prms': [6.23741017, -2.1430682, 4.59332686, 20.0],
        'opt_wts': np.array([10.        , -3.26057123]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([0, 5]),
        'opt_kpeaks': np.array([ 4.73623985, 21.59707569])
    }, # best chosen fit for sOFF 15 Hz
    "tOFF_TF4" : {
        'sopt_msg': ('After ', '14', 'function evaluations, TNC returned:',
            'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 14,
        'sopt_res': np.array([ 3.34598077, -2.18198978, 29.15797185, 59.36268031]),
        'init_prms': [3.5, -2.0, 30.0, 60.0],
        'opt_wts': np.array([ 3.34598077, -2.18198978]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([ 0, 15]),
        'opt_kpeaks': np.array([29.15797185, 59.36268031])
    }, # best chosen fit for tOFF 4 Hz
    "tOFF_TF8" : {
        'sopt_msg': ('After ', '20', 'function evaluations, TNC returned:',
                     'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 20,
        'sopt_res': np.array([ 4.21848979, -2.40498983,  8.52464523, 23.24681671]),
        'init_prms': [4.222, -2.404, 8.545, 23.019],
        'opt_wts': np.array([ 4.21848979, -2.40498983]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([0, 0]),
        'opt_kpeaks': np.array([ 8.52464523, 23.24681671])
    }, # best chosen fit for tOFF 8 Hz
    "tOFF_TF15" : {
        'sopt_msg': ('After ', '8', 'function evaluations, TNC returned:',
            'Converged (|f_n-f_(n-1)| ~= 0)'),
        'sopt_nf': 8,
        'sopt_res': np.array([ 3.44166036, -2.11559948,  8.2697336 , 19.99148791]),
        'init_prms': [3.44215357, -2.11509939, 8.27421573, 20.0],
        'opt_wts': np.array([ 3.44166036, -2.11559948]),
        'prms_bds': [(1.0, 10.0), (-10.0, -0.05), (2, 100), (2, 200)],
        'opt_delays': np.array([0, 0]),
        'opt_kpeaks': np.array([ 8.2697336 , 19.99148791])
    }, # best chosen fit for tOFF 15 Hz
}


def lognorm_ppf(
    x: npt.NDArray[np.floating[t.Any]],
    shape: float,
    loc: float=0,
    scale: float=1.0
):
    # definition from wikipedia (quantile)
    return scale * exp(sqrt(2 * shape**2) * erfinv(2 * x - 1)) + loc


def delta_theta_cdf(intercept: float, d_theta: float):
    B1 = intercept
    # B1 = 2.0 / (1.0 + Q)
    Q = 2.0 / B1 - 1.0
    B2 = B1 * Q
    G = (B2 - B1) / 90.0
    norm = 90 * (B1 + B2)  # total area for normalization
    x = d_theta - 90
    if d_theta < 0:
        raise Exception("d_theta must be >= 0, but was {}".format(d_theta))
    elif d_theta < 90:
        # analytical integration of the pdf to get this cdf
        return (0.5 * G * x**2 + B2 * x) / norm + 0.5
    elif d_theta <= 180:
        return (-0.5 * G * x**2 + B2 * x) / norm + 0.5
    else:
        raise Exception("d_theta must be <= 180, but was {}".format(d_theta))


def compute_pair_type_parameters(
    source_type: str,
    target_type: str,
    cc_props: dict[str, t.Any] 
):
    """Takes in two strings for the source and target type. It determined the connectivity parameters needed based on
    distance dependence and orientation tuning dependence and returns a dictionary of these parameters. A description
    of the calculation and resulting formulas used herein can be found in the accompanying documentation. Note that the
    source and target can be the same as this function works on cell TYPES and not individual nodes. The first step of
    this function is getting the parameters that determine the connectivity probabilities reported in the literature.
    From there the calculation proceed based on adapting these values to our model implementation.

    :param source_type: string of the cell type that will be the source (pre-synaptic)
    :param target_type: string of the cell type that will be the targer (post-synaptic)
    :return: dictionary with the values to be used for distance dependent connectivity
             and orientation tuning dependent connectivity (when applicable, else nans populate the dictionary).
    """
    # A_literature is different for every source-target pair and was estimated from the literature.
    A_literature = cc_props["A_literature"][0]  # TODO: remove [0] after fixing the json file

    # R0 read from the dictionary, but setting it now at 75um for all cases but this allows us to change it
    R0 = cc_props["R0"]

    # Sigma is measure from the literature or internally at the Allen Institute
    sigma = cc_props["sigma"]

    # Gaussian equation was intergrated to and solved to calculate new A_new. See accompanying documentation.
    if cc_props["is_pmax"] == 1:
        A_new = A_literature
    else:
        A_new = A_literature / ((sigma / R0) ** 2 * (1 - np.exp(-((R0 / sigma) ** 2))))

    # Due to the measured values in the literature being from multiple sources and approximations that were
    # made by us and the literature (for instance R0 = 75um and sigma from the literature), we found that it is
    # possible for A_new to go slightly above 1.0 in which case we rescale it to 1.0. We confirmed that if this
    # does happen, it is for a few cases and is not much higher than 1.0.
    if A_new > 1.0:
        # print('WARNING: Adjusted calculated probability based on distance dependence is coming out to be ' \
        #       'greater than 1 for ' + source_type + ' and ' + target_type + '. Setting to 1.0')
        A_new = 1.0

    # ### To include orientation tuning ####
    # Many cells will show orientation tuning and the relative different in orientation tuning angle will influence
    # probability of connections as has been extensively report in the literature. This is modeled here with a linear
    # where B in the largest value from 0 to 90 (if the profile decays, then B is the intercept, if the profile
    # increases, then B is the value at 90). The value of G is the gradient of the curve.
    # The calculations and explanation can be found in the accompanying documentation with this code.

    # Extract the values from the dictionary for B, the maximum value and G the gradient
    B_ratio = cc_props["B_ratio"]
    B_ratio = np.nan if B_ratio is None else B_ratio

    # Check if there is orientation dependence in this source-target pair type. If yes, then a parallel calculation
    # to the one done above for distance dependence is made though with the assumption of a linear profile.
    if not np.isnan(B_ratio):
        # The scaling for distance and orientation must remain less than 1 which is calculated here and reset
        # if it is greater than one. We also ensure that the area under the p(delta_phi) curve is always equal
        # to one (see documentation). Hence the desired ratio by the user may not be possible, in which case
        # an warning message appears indicating the new ratio used. In the worst case scenario the line will become
        # horizontal (no orientation tuning) but will never "reverse" slopes.

        # B1 is the intercept which occurs at (0, B1)
        # B2 is the value when delta_phi equals 90 degree and hence the point (90, B2)
        B1 = 2.0 / (1.0 + B_ratio)
        B2 = B_ratio * B1

        AB = A_new * max(B1, B2)
        if AB > 1.0:
            if B1 >= B2:
                B1_new = 1.0 / A_new
                delta = B1 - B1_new
                B1 = B1_new     # pyright: ignore[reportConstantRedefinition]
                B2 = B2 + delta # pyright: ignore[reportConstantRedefinition]
            elif B2 > B1:
                B2_new = 1.0 / A_new
                delta = B2 - B2_new
                B2 = B2_new # pyright: ignore[reportConstantRedefinition]
                B1 = B1 + delta # pyright: ignore[reportConstantRedefinition]

            B_ratio = B2 / B1
            print(
                "WARNING: Could not satisfy the desired B_ratio "
                + "(probability of connectivity would become "
                + "greater than one in some cases). Rescaled and now for "
                + source_type
                + " --> "
                + target_type
                + " the ratio is set to: ",
                B_ratio,
            )

        G = (B2 - B1) / 90.0

    # If there is no orientation dependent, record this by setting the intercept to Not a Number (NaN).
    else:
        B1 = np.NaN # pyright: ignore[reportConstantRedefinition]
        G = np.NaN # pyright: ignore[reportConstantRedefinition]

    # Return the dictionary. Note, the new values are A_new and intercept. The rest are from CC_prob_dict.
    return {
        "A_new": A_new,
        "sigma": sigma,
        "gradient": G,
        "intercept": B1,
        "nsyn_range": cc_props["nsyn_range"],
        "src_ei": cc_props["pre_ei"],   # for Rossi correction
        "trg_ei": cc_props["post_ei"],  # for Rossi
    }


def connect_cells(
    _sources,
    target,
    params: dict[str, t.Any],
    source_nodes,
    core_radius: float
):
    """This function determined which nodes are connected based on the parameters in the dictionary params. The
    function iterates through every cell pair when called and hence no for loop is seen iterating pairwise
    although this is still happening.

    By iterating though every cell pair, a decision is made whether or not two cells are connected and this
    information is returned by the function. This function calculates these probabilities based on the distance between
    two nodes and (if applicable) the orientation tuning angle difference.

    :param sid: the id of the source node
    :param source: the attributes of the source node
    :param tid: the id of the target node
    :param target: the attributes of the target node
    :param params: parameters dictionary for probability of connection (see function: compute_pair_type_parameters)
    :return: if two cells are deemed to be connected, the return function automatically returns the source id
             and the target id for that connection. The code further returns the number of synapses between
             those two neurons
    """
    # special handing of the empty sources
    if source_nodes.empty:
        # since there is no sources, no edges will be created.
        print("Warning: no sources for target: {}".format(target.node_id))
        return []

    # TODO: remove list comprehension
    # sources_x = np.array([s["x"] for s in sources])
    # sources_z = np.array([s["z"] for s in sources])
    # sources_tuning_angle = [s["tuning_angle"] for s in sources]
    sources_x = np.array(source_nodes["x"])
    sources_z = np.array(source_nodes["z"])
    sources_tuning_angle = np.array(source_nodes["tuning_angle"])

    # Get target id
    tid = target.node_id
    # if tid % 1000 == 0:
    #     print('target {}'.format(tid))

    # size of target cell (total syn number) will modulate connection probability
    target_size = target["target_sizes"]
    target_pop_mean_size = target["nsyn_size_mean"]

    # Read parameter values needed for distance and orientation dependence
    A_new = params["A_new"]
    sigma = params["sigma"]
    gradient = params["gradient"]
    intercept = params["intercept"]
    # nsyn_range = params["nsyn_range"]

    # Calculate the intersomatic distance between the current two cells (in 2D - not including depth)
    intersomatic_distance = np.sqrt(
        (sources_x - target["x"]) ** 2 + (sources_z - target["z"]) ** 2
    )

    # For Rossi correction. The distance dependence is asymmetrical in the X-Z plane and depends on the target neuron
    # orientation preference.
    intersomatic_x = sources_x - target["x"]
    intersomatic_z = sources_z - target["z"]
    intersomatic_xz = [
        [delta_x, delta_z] for delta_x, delta_z in zip(intersomatic_x, intersomatic_z)
    ]
    if np.sqrt(target["x"] ** 2 + target["z"] ** 2) > (core_radius * 1.5):
        Rossi_displacement = 0.0
    else:
        Rossi_displacement = (
            50.0  # displacement of presynaptic soma distribution centroid
        )
    Rossi_scaling = 1.5  # scaling of major/minor axes of covariance

    Rossi_theta = np.radians(target["tuning_angle"])

    if params["src_ei"] == "i":
        Rossi_mean = [
            Rossi_displacement * np.cos(Rossi_theta),
            Rossi_displacement * np.sin(Rossi_theta),
        ]
        cov_ = np.array([[(sigma) ** 2, 0.0], [0.0, (sigma) ** 2]])
    else:
        Rossi_mean = [
            -Rossi_displacement * np.cos(Rossi_theta),
            -Rossi_displacement * np.sin(Rossi_theta),
        ]
        cov_ = np.array(
            [[(sigma / Rossi_scaling) ** 2, 0.0], [0.0, (sigma * Rossi_scaling) ** 2]]
        )

    c, s = np.cos(Rossi_theta), np.sin(Rossi_theta)
    R = np.array(((c, -s), (s, c)))
    Rossi_cov = R @ cov_ @ R.transpose()

    Rossi_mvNorm = multivariate_normal(Rossi_mean, Rossi_cov)  # pyright: ignore[reportArgumentType] 

    # if target.node_id % 10000 == 0:
    #     print("Working on tid: ", target.node_id)

    # Check if there is orientation dependence
    if not np.isnan(gradient):
        # Calculate the difference in orientation tuning between the cells
        delta_orientation = np.array(sources_tuning_angle, dtype=float) - float(
            target["tuning_angle"]
        )

        # For OSI, convert to quadrant from 0 - 90 degrees
        delta_orientation = abs(abs(abs(180.0 - abs(delta_orientation)) - 90.0) - 90.0)

        # Calculate the probability two cells are connected based on distance and orientation
        # p_connect = (
        #    A_new
        #    * np.exp(-((intersomatic_distance / sigma) ** 2))
        #    * (intercept + gradient * delta_orientation)
        # )

        # using Rossi:
        p_connect = (
            A_new
            * Rossi_mvNorm.pdf(intersomatic_xz)
            / Rossi_mvNorm.pdf(Rossi_mean)
            * (intercept + gradient * delta_orientation)
        )

    # If no orientation dependence
    else:
        # Calculate the probability two cells are connection based on distance only
        # p_connect = A_new * np.exp(-((intersomatic_distance / sigma) ** 2))

        # using Rossi:
        p_connect = (
            A_new * Rossi_mvNorm.pdf(intersomatic_xz) / Rossi_mvNorm.pdf(Rossi_mean)
        )

    # # Sanity check warning
    # if p_connect > 1:
    #    print(
    #        " WARNING: p_connect is greater that 1.0 it is: "
    #        + str(p_connect)
    #    )

    # If not the same cell (no self-connections)
    if 0.0 in intersomatic_distance:
        p_connect[np.where(intersomatic_distance == 0.0)[0][0]] = 0

    # Connection p proportional to target cell synapse number relative to population average:
    p_connect = p_connect * target_size / target_pop_mean_size

    # If p_connect > 1 set to 1:
    p_connect[p_connect > 1] = 1

    # Decide which cells get a connection based on the p_connect value calculated
    p_connected = np.random.binomial(1, p_connect)

    # Synapse number only used for calculating numbers of "leftover" syns to assign as background;
    #    N_syn_ will be added through 'add_properties'
    # p_connected[p_connected == 1] = 1

    # p_connected[p_connected == 1] = np.random.randint(
    #    nsyn_range[0], nsyn_range[1], len(p_connected[p_connected == 1])
    # )

    # TODO: remove list comprehension
    nsyns_ret = [Nsyn if Nsyn != 0 else None for Nsyn in p_connected]
    return nsyns_ret


def syn_weight_by_experimental_distribution(
    source: dict[str, t.Any],
    target: dict[str, t.Any],
    src_type: str,  # pyright: ignore[reportUnusedParameter]
    trg_type: str,  # pyright: ignore[reportUnusedParameter]
    src_ei: str,
    trg_ei: str,
    PSP_correction: float,
    PSP_lognorm_shape: float,
    PSP_lognorm_scale: float,
    connection_params: dict[str, t.Any],
    # delta_theta_dist,
):
    # src_ei = "e" if src_type.startswith("e") or src_type.startswith("LIFe") else "i"
    # trg_ei = "e" if trg_type.startswith("e") or trg_type.startswith("LIFe") else "i"
    src_tuning = source["tuning_angle"]
    tar_tuning = target["tuning_angle"]

    #
    if PSP_lognorm_shape < target["nsyn_size_shape"]:
        weight_shape = 0.001
    else:
        weight_shape = sqrt(PSP_lognorm_shape**2 - target["nsyn_size_shape"] ** 2)
    weight_scale = exp(
        log(PSP_lognorm_scale)
        + log(target["nsyn_size_scale"])
        - log(target["nsyn_size_mean"])
    )

    randomizing_factor = 1.0
    if (
        src_ei == "e"
        and trg_ei == "e"
        and (not np.isnan(connection_params["gradient"]))
    ):
        # For e-to-e, there is a non-uniform distribution of delta_orientations.
        # These need to be ordered and mapped uniformly over [0,1] using the cdf:
        tuning_rnd = 0.0
        delta_tuning_180 = np.abs(
            np.abs(np.mod(np.abs(tar_tuning - src_tuning + tuning_rnd), 360.0) - 180.0)
            - 180.0
        )
        orient_temp = 1 - delta_theta_cdf(
            connection_params["intercept"], delta_tuning_180
        )
        # Clamp the orientation to avoid extremes
        orient_temp = np.clip(orient_temp, 0.001, 0.999)
        #
        syn_weight = lognorm_ppf(orient_temp, weight_shape, loc=0, scale=weight_scale)
        n_syns_ = 1
    elif (src_ei == "e" and trg_ei == "i") or (src_ei == "i" and trg_ei == "e"):
        # If there was no like-to-like connection rule for the population, we can use
        # delta_orientation directly with the PPF
        # adds some randomization to like-to-like and avoids 0-degree delta
        tuning_rnd = float(np.random.randn(1) * 5) * randomizing_factor

        delta_tuning_180 = np.abs(
            np.abs(np.mod(np.abs(tar_tuning - src_tuning + tuning_rnd), 360.0) - 180.0)
            - 180.0
        )
        orient_temp = 1 - (delta_tuning_180 / 180)
        # Clamp the orientation to avoid extremes
        orient_temp = np.clip(orient_temp, 0.001, 0.999)
        #
        syn_weight = lognorm_ppf(orient_temp, weight_shape, loc=0, scale=weight_scale)
        n_syns_ = 1
    elif src_ei == "i" and trg_ei == "i":
        # If there was no like-to-like connection rule for the population, we can use
        # delta_orientation directly with the PPF
        # adds some randomization to like-to-like and avoids 0-degree delta
        tuning_rnd = float(np.random.randn(1) * 10) * randomizing_factor
        delta_tuning_180 = np.abs(
            np.abs(np.mod(np.abs(tar_tuning - src_tuning + tuning_rnd), 360.0) - 180.0)
            - 180.0
        )
        orient_temp = 1 - (delta_tuning_180 / 180)
        # Clamp the orientation to avoid extremes
        orient_temp = np.clip(orient_temp, 0.001, 0.999)
        #
        syn_weight = lognorm_ppf(orient_temp, weight_shape, loc=0, scale=weight_scale)
        n_syns_ = 1
    else:
        # If there was no like-to-like connection rule for the population, we can use
        # delta_orientation directly with the PPF

        # adds some randomization to like-to-like and avoids 0-degree delta
        tuning_rnd = float(np.random.randn(1) * 5) * randomizing_factor

        delta_tuning_180 = np.abs(
            np.abs(np.mod(np.abs(tar_tuning - src_tuning + tuning_rnd), 360.0) - 180.0)
            - 180.0
        )
        orient_temp = 1 - (delta_tuning_180 / 180)
        # Clamp the orientation to avoid extremes
        orient_temp = np.clip(orient_temp, 0.001, 0.999)
        #
        syn_weight = lognorm_ppf(orient_temp, weight_shape, loc=0, scale=weight_scale)
        n_syns_ = 1

    syn_weight = (
        syn_weight
        * target["nsyn_size_mean"]
        / (PSP_correction * target["target_sizes"])
    )
    return syn_weight, n_syns_


def generate_random_positions(
        N: int, layer_range: tuple[int, int], radial_range: tuple[int, int]
):
    radius_outer = radial_range[1]
    radius_inner = radial_range[0]

    phi = 2.0 * np.pi * np.random.random([N])
    r = np.sqrt(
        (radius_outer ** 2 - radius_inner ** 2) * np.random.random([N])
        + radius_inner ** 2
    )
    x = r * np.cos(phi)
    z = r * np.sin(phi)

    layer_start = layer_range[0]
    layer_end = layer_range[1]
    # Generate N random z values.
    y = (layer_end - layer_start) * np.random.random([N]) + layer_start

    positions = np.column_stack((x, y, z))

    return positions


def generate_positions_grids(
    N: int, x_grids: int, y_grids: int, x_len: float, y_len: float
):
    widthPerTile = x_len / x_grids
    heightPerTile = y_len / y_grids

    X = np.zeros(N * x_grids * y_grids)
    Y = np.zeros(N * x_grids * y_grids)

    counter = 0
    for i in range(x_grids):
        for j in range(y_grids):
            x_tile = np.random.uniform(i * widthPerTile, (i + 1) * widthPerTile, N)
            y_tile = np.random.uniform(j * heightPerTile, (j + 1) * heightPerTile, N)
            X[counter * N : (counter + 1) * N] = x_tile
            Y[counter * N : (counter + 1) * N] = y_tile
            counter = counter + 1
    return np.column_stack((X, Y))


def get_filter_spatial_size(
    N: int, X_grids: int, Y_grids: int, size_range: list[int]
):
    spatial_sizes = np.zeros(N * X_grids * Y_grids)
    counter = 0
    for i in range(X_grids):
        for j in range(Y_grids):
            if len(size_range) == 1:
                sizes = np.ones(N) * size_range[0]
            else:
                sizes = np.random.triangular(
                    size_range[0], size_range[0] + 1, size_range[1], N
                )
            spatial_sizes[counter * N : (counter + 1) * N] = sizes
            counter = counter + 1
    return spatial_sizes


def select_bkg_sources(
    sources: list[t.Any], _target: t.Any, n_syns: int, n_conn:int
):
    # draw n_conn connections randomly from the background sources.
    # n_syns is the number of synapses per connection
    # n_conn is the number of connections to draw
    n_unit = len(sources)
    # select n_conn units randomly
    selected_units = np.random.choice(n_unit, size=n_conn, replace=False)
    nsyns_ret = np.zeros(n_unit, dtype=int)
    nsyns_ret[selected_units] = n_syns
    nsyns_ret = list(nsyns_ret)
    # getting back to list
    nsyns_ret = [None if n == 0 else n for n in nsyns_ret]
    return nsyns_ret


def lgn_synaptic_weight_rule(
    source: t.Any,
    target: dict[str, t.Any],
    base_weight:float,
    mean_size: float
):
    return base_weight * mean_size / target["target_sizes"]



def write_bkg(
    output_filename:str | pathlib.Path,
    n_neu:int=1,
    duration:float=3.0,
    binsize:float=2.5e-4,
    rate:int=1000,
    seed:int=0
):
    # time units are seconds, (inverse: Hz)
    nbins = int(duration / binsize)
    np.random.seed(seed)
    spike_bools = np.random.random([n_neu, nbins]) < (rate * binsize)
    where = np.where(spike_bools)
    # spikes_time = (np.where(spike_bools)[0] + 1) * binsize  # to avoid 0, still second
    timestamps = (where[1] + 1) * binsize
    # nids = np.zeros_like(spikes_time, dtype=np.uint)
    nids = where[0]

    # save
    out_file = h5py.File(output_filename, "w")
    # out_file["spikes/gids"] = nids
    # add some random value to avoid the bad time stamps.
    # out_file["spikes/timestamps"] = timestamps * 1000 + 0.01  # in ms
    # let's use gzip level 6.
    out_file.create_dataset(
        "spikes/gids",
        data=nids,
        compression="gzip",
        compression_opts=6,
        shuffle=True,
    )
    out_file.create_dataset(
        "spikes/timestamps",
        data=timestamps * 1000 + 0.01,
        compression="gzip",
        compression_opts=6,
        shuffle=True,
    )
    out_file.close()
    return 0


def write_regular_bkg(output_fpath: pathlib.Path, duration: float=1.0, interval: float=0.1) -> None:
    spikes_time = np.linspace(interval, duration, int(duration / interval))
    nids = np.zeros_like(spikes_time, dtype=np.uint)

    # save
    out_file = h5py.File(output_fpath, "w")
    out_file["spikes/gids"] = nids
    out_file["spikes/timestamps"] = spikes_time * 1000  # in ms
    out_file.close()


def generate_bkg_spikes(base_dir: str, bkg_file: str, bkg_dir_name : str="bkg"):
    # try to write the bkg (let's make all of them)
    # basedir = "small"
    bkg_dir = pathlib.Path(base_dir, bkg_dir_name)
    bkg_dir.mkdir(parents=True, exist_ok=True)
    # read the bkg file and check the number of bkg units.
    # bkg_file = f"{basedir}/network/bkg_nodes.h5"
    # open the file and get the number of bkg units
    with h5py.File(bkg_file, "r") as hfptr:
        n_neu = hfptr["nodes"]["bkg"]["node_id"].shape[0]  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]

    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_1kHz_100s.h5"
    # write_bkg(bkg_name, n_neu, duration=100.0)
    bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_1kHz_10s.h5")
    write_bkg(bkg_name, n_neu, duration=10.0)
    bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_1kHz_3s.h5")
    write_bkg(bkg_name, n_neu)

    # 250 Hz (new default for 100 background units)
    bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_250Hz_10s.h5")
    write_bkg(bkg_name, n_neu, rate=250, duration=10.0)
    bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_250Hz_3s.h5")
    write_bkg(bkg_name, n_neu, rate=250, duration=3.0)
    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_100Hz_10s.h5")
    # write_bkg(bkg_name, n_neu, rate=100, duration=10.0)
    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_100Hz_3s.h5")
    # write_bkg(bkg_name, n_neu, rate=100, duration=3.0)
    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_2kHz_10s.h5")
    # write_bkg(bkg_name, n_neu, rate=2000, duration=10.0)
    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_2kHz_3s.h5")
    # write_bkg(bkg_name, n_neu, rate=2000, duration=3.0)

    # with 'full', all the bins are filled with spikes. i.e. no fluctuations.
    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_full_100s.h5")
    # write_bkg(bkg_name, n_neu, rate=4000, duration=100.0)
    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_full_30s.h5")
    # write_bkg(bkg_name, n_neu, rate=4000, duration=30.0)
    bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_full_10s.h5")
    write_bkg(bkg_name, n_neu, rate=4000, duration=10.0)
    bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_full_3s.h5")
    write_bkg(bkg_name, n_neu, rate=4000, duration=3.0)
    # these require high resolution simulations
    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_5kHz_3s.h5"
    # write_bkg(bkg_name, n_neu, rate=5000, binsize=1.0e-5, duration=3.0)
    # bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_5kHz_10s.h5"
    # write_bkg(bkg_name, n_neu, rate=5000, binsize=1.0e-5, duration=10.0)

    # regular bkg
    bkg_name = pathlib.Path(bkg_dir, "bkg_spikes_regular_1s.h5")
    write_regular_bkg(bkg_name)

    # for the 8 direction stimuli (10 repetition)
    start_seed = 381583
    for i in range(8):
        for j in range(10):
            seed = start_seed + i * 10 + j
            dirname = pathlib.Path(base_dir, "bkg_8dir_10trials", f"angle{i*45}_trial{j}")
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            # write_bkg(pathlib.Path(dirname, "bkg_spikes_1kHz_3s.h5"), n_neu, seed=seed)
            write_bkg(pathlib.Path(dirname, "bkg_spikes_250Hz_3s.h5"), n_neu, rate=250, seed=seed)
            # write_bkg(pathlib.Path(dirname, "bkg_spikes_2kHz_3s.h5"), n_neu, rate=2000, seed=seed)
            # write_bkg(pathlib.Path(dirname, "bkg_spikes_full_3s.h5"), n_neu, rate=4000, seed=seed)


def convert_ctdb_models_to_nest(input_dir:str, output_dir:str):
    cfg_fnames = os.listdir(input_dir)
    for fxname in cfg_fnames:
        in_fxname = pathlib.Path(input_dir, fxname)
        with open(in_fxname) as ifx:
            cfg_db = json.load(ifx)
        cvtx = glif_utils.converter_builtin("nest:glif_lif_asc_psc", cfg_db)
        if not cvtx:
            continue
        n_dict = cvtx[1]
        #del n_dict["tau_syn"]
        for kx, valx in n_dict.items():
            if valx.__class__ == np.ndarray:
                n_dict[kx] = list(valx) # pyright: ignore[reportArgumentType]
        out_fname = pathlib.Path(output_dir, fxname)
        with open(out_fname, "w") as ofx:
            json.dump(n_dict, ofx, indent=4)


def get_filter_temporal_params(N:int, X_grids:int, Y_grids:int, model:str):
    # Total number of cells
    N_total = N * X_grids * Y_grids

    # Jitter parameters
    jitter : float = 0.025
    lower_jitter : float = 1 - jitter
    upper_jitter : float = 1 + jitter

    # Directory of pickle files with saved parameter values
    # basepath = "base_props/lgn_fitted_models/"

    # For two-subunit filter (sONsOFF and sONtOFF)
    sOFF_prs = LGN_BEST_FIT["sOFF_TF4"] # best chosen fit for sOFF 4 Hz
    tOFF_prs = LGN_BEST_FIT["tOFF_TF8"] # best chosen fit for tOFF 8 Hz
    sON_prs = LGN_BEST_FIT["sON_TF4"] # best chosen fit for sON 4 Hz

    # Choose cell type and temporal frequency
    if model == "sONsOFF_001":
        kpeaks : np.array = sOFF_prs["opt_kpeaks"]
        kpeaks_dom_0 = np.random.uniform(
            lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total
        )
        kpeaks_dom_1 = np.random.uniform(
            lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total
        )
        kpeaks = sON_prs["opt_kpeaks"]
        kpeaks_non_dom_0 = np.random.uniform(
            lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total
        )
        kpeaks_non_dom_1 = np.random.uniform(
            lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total
        )

        wts : np.array = sOFF_prs["opt_wts"]
        wts_dom_0 = np.random.uniform(
            lower_jitter * wts[0], upper_jitter * wts[0], N_total
        )
        wts_dom_1 = np.random.uniform(
            lower_jitter * wts[1], upper_jitter * wts[1], N_total
        )
        wts = sON_prs["opt_wts"]
        wts_non_dom_0 = np.random.uniform(
            lower_jitter * wts[0], upper_jitter * wts[0], N_total
        )
        wts_non_dom_1 = np.random.uniform(
            lower_jitter * wts[1], upper_jitter * wts[1], N_total
        )

        delays : np.array = sOFF_prs["opt_delays"]
        delays_dom_0 = np.random.uniform(
            lower_jitter * delays[0], upper_jitter * delays[0], N_total
        )
        delays_dom_1 = np.random.uniform(
            lower_jitter * delays[1], upper_jitter * delays[1], N_total
        )
        delays = sON_prs["opt_delays"]
        delays_non_dom_0 = np.random.uniform(
            lower_jitter * delays[0], upper_jitter * delays[0], N_total
        )
        delays_non_dom_1 = np.random.uniform(
            lower_jitter * delays[1], upper_jitter * delays[1], N_total
        )

        sf_sep = 6.0
        sf_sep = np.random.uniform(
            lower_jitter * sf_sep, upper_jitter * sf_sep, N_total
        )
        tuning_angles = np.random.uniform(0, 360.0, N_total)

    elif model == "sONtOFF_001":
        kpeaks = tOFF_prs["opt_kpeaks"]
        kpeaks_dom_0 = np.random.uniform(
            lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total
        )
        kpeaks_dom_1 = np.random.uniform(
            lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total
        )
        kpeaks = sON_prs["opt_kpeaks"]
        kpeaks_non_dom_0 = np.random.uniform(
            lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total
        )
        kpeaks_non_dom_1 = np.random.uniform(
            lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total
        )

        wts = tOFF_prs["opt_wts"]
        wts_dom_0 = np.random.uniform(
            lower_jitter * wts[0], upper_jitter * wts[0], N_total
        )
        wts_dom_1 = np.random.uniform(
            lower_jitter * wts[1], upper_jitter * wts[1], N_total
        )
        wts = sON_prs["opt_wts"]
        wts_non_dom_0 = np.random.uniform(
            lower_jitter * wts[0], upper_jitter * wts[0], N_total
        )
        wts_non_dom_1 = np.random.uniform(
            lower_jitter * wts[1], upper_jitter * wts[1], N_total
        )

        delays = tOFF_prs["opt_delays"]
        delays_dom_0 = np.random.uniform(
            lower_jitter * delays[0], upper_jitter * delays[0], N_total
        )
        delays_dom_1 = np.random.uniform(
            lower_jitter * delays[1], upper_jitter * delays[1], N_total
        )
        delays = sON_prs["opt_delays"]
        delays_non_dom_0 = np.random.uniform(
            lower_jitter * delays[0], upper_jitter * delays[0], N_total
        )
        delays_non_dom_1 = np.random.uniform(
            lower_jitter * delays[1], upper_jitter * delays[1], N_total
        )

        sf_sep = 4.0
        sf_sep = np.random.uniform(
            lower_jitter * sf_sep, upper_jitter * sf_sep, N_total
        )
        tuning_angles = np.random.uniform(0, 360.0, N_total)

    else:
        # cell_type = model[0 : model.find("_")]  #'sON'  # 'tOFF'
        # tf_str = model[model.find("_") + 1 :]

        # # Load pickle file containing params for optimized temporal kernel, it it exists
        # file_found = 0
        # for fname in os.listdir(basepath):
        #     if os.path.isfile(os.path.join(basepath, fname)):
        #         pkl_savename = os.path.join(basepath, fname)
        #         if (
        #             tf_str in pkl_savename.split("_")
        #             and pkl_savename.find(cell_type) >= 0
        #             and pkl_savename.find(".pkl") >= 0
        #         ):
        #             file_found = 1
        #             print(pkl_savename)
        #             filt_file = pkl_savename

        # if file_found != 1:
        #     print("File not found: Filter was not optimized for this sub-class")

        # # savedata_dict = pickle.load(open(filt_file, 'rb'))
        # savedata_dict = open_pickle(filt_file)  # pickle.load(open(filt_file, 'rb'))
        params_dict = LGN_BEST_FIT[model]

        kpeaks = params_dict["opt_kpeaks"]
        kpeaks_dom_0 = np.random.uniform(
            lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total
        )
        kpeaks_dom_1 = np.random.uniform(
            lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total
        )
        kpeaks_non_dom_0 = np.nan * np.zeros(N_total)
        kpeaks_non_dom_1 = np.nan * np.zeros(N_total)

        wts = params_dict["opt_wts"]
        wts_dom_0 = np.random.uniform(
            lower_jitter * wts[0], upper_jitter * wts[0], N_total
        )
        wts_dom_1 = np.random.uniform(
            lower_jitter * wts[1], upper_jitter * wts[1], N_total
        )
        wts_non_dom_0 = np.nan * np.zeros(N_total)
        wts_non_dom_1 = np.nan * np.zeros(N_total)

        delays = params_dict["opt_delays"]
        delays_dom_0 = np.random.uniform(
            lower_jitter * delays[0], upper_jitter * delays[0], N_total
        )
        delays_dom_1 = np.random.uniform(
            lower_jitter * delays[1], upper_jitter * delays[1], N_total
        )
        delays_non_dom_0 = np.nan * np.zeros(N_total)
        delays_non_dom_1 = np.nan * np.zeros(N_total)

        sf_sep = np.nan * np.zeros(N_total)
        tuning_angles = np.nan * np.zeros(N_total)

    return np.column_stack(
        (
            kpeaks_dom_0,
            kpeaks_dom_1,
            wts_dom_0,
            wts_dom_1,
            delays_dom_0,
            delays_dom_1,
            kpeaks_non_dom_0,
            kpeaks_non_dom_1,
            wts_non_dom_0,
            wts_non_dom_1,
            delays_non_dom_0,
            delays_non_dom_1,
            tuning_angles,
            sf_sep,
        )
    )

# @njit
def within_ellipse(
    x: npt.NDArray[np.floating[t.Any]],
    y: npt.NDArray[np.floating[t.Any]],
    tuning_angle: float | None,
    e_x: float, e_y: float,
    e_cos: float, e_sin: float,
    e_a: float, e_b: float
):
    """check if x, y are within the ellipse"""
    x0 = x - e_x
    y0 = y - e_y
    if tuning_angle is None:
        x_rot = x0
        y_rot = y0
    else:
        x_rot = x0 * e_cos - y0 * e_sin
        y_rot = x0 * e_sin + y0 * e_cos
    return ((x_rot / e_a) ** 2 + (y_rot / e_b) ** 2) <= 1.0


def calculate_subunit_probs(cell_TF, tf_list):
    tf_array = np.array(tf_list)
    tf_sum = np.sum(abs(cell_TF - tf_array))
    p = (1 - abs(cell_TF - tf_array) / tf_sum) / (len(tf_array) - 1)
    return p


# orientation comparator
def delta_ori(angle: float):
    return np.abs(np.abs(angle - 90) % 180 - 90)


def gaussian_probability(x, sigma):
    return np.exp(-(x**2 / (2 * sigma**2)))


def pick_from_probs(n: int, prob_dist:npt.NDArray[np.floating[t.Any]]):
    try:
        # pick n item based on prob_dist and return index of the choice
        return np.random.choice(
            list(range(len(prob_dist))), 
            size=n,
            replace=False,
            p=None if np.isnan(np.sum(prob_dist)) else prob_dist
        )
    except ValueError as vex:
        print(prob_dist)
        raise vex


def select_lgn_sources_powerlaw(sources, target, lgn_mean, lgn_nodes):
    target_id = target.node_id
    pop_name = target["pop_name"]

    if target_id % 250 == 0:
        print("connection LGN cells to V1 cell #", target_id)

    # the coordinates are already in visual field space, so simple multiplication is all you need.
    x_position_lin_degrees = target["x"] * 0.07
    y_position_lin_degrees = target["z"] * 0.04

    # center of the visual RF
    vis_x = lgn_mean[0] + x_position_lin_degrees
    vis_y = lgn_mean[1] + y_position_lin_degrees
    rf_center = vis_x + 1j * vis_y

    tuning_angle = float(target["tuning_angle"])
    tuning_angle = None if math.isnan(tuning_angle) else tuning_angle
    testing = False
    cell_ignore_unit = None
    rf_shift_vector = np.exp(0j)
    if tuning_angle is not None:
        tuning_angle_rad = tuning_angle / 180.0 * math.pi
        if testing:
            rf_shift_vector = -np.exp(1j * tuning_angle_rad)  # tentatively flipping
            probability_sON = 1.0  # for testing purpose fix later
        else:
            rf_shift_vector = np.exp(1j * tuning_angle_rad)  # using complex expression
            probability_sON = LGN_PARAMS.loc[pop_name, "sON_ratio"]
        if np.random.random() < probability_sON:
            cell_ignore_unit = "sOFF"  # sON cell. ignore sOFF
        else:
            cell_ignore_unit = "sON"
            rf_shift_vector = -rf_shift_vector  # This will be flipped.

    # not very comfortable with this.
    cell_TF = np.random.poisson(LGN_PARAMS.loc[pop_name, "poisson_parameter"])
    while cell_TF <= 0:
        cell_TF = np.random.poisson(LGN_PARAMS.loc[pop_name, "poisson_parameter"])

    subunit_freqs = {"sON": [1, 2, 4, 8], "sOFF": [1, 2, 4, 8, 15], "tOFF": [4, 8, 15]}
    # calculate probability for each subunit types separately

    # start calculating the relative probability for each candidate neuron
    # make a candidate pool (circular)

    # circle with radius 40 centered at vis_x, vis_y
    big_circle = (vis_x, vis_y, 1.0, 0.0, 40, 40)
    in_circle = within_ellipse(
        np.array(lgn_nodes["x"]),
        np.array(lgn_nodes["y"]),
        tuning_angle,
        *big_circle
    )

    lgn_circle = lgn_nodes[in_circle]
    # if there is no candidate LGN cells, return with no connectinos.
    if lgn_circle.empty:
        # print("Warning: no candidate LGN cells for V1 cell id: ", target_id)
        return [None] * len(lgn_nodes)

    # RF center of LGN cell as a complex number
    lgn_complex = np.array(lgn_circle["x"] + 1j * lgn_circle["y"])

    # shift sustained and transient units
    if testing:
        shift = 2.5  # amount to shift the RF
    else:
        shift = 2.5  # amount to shift the RF

    lgn_complex += np.array(
        lgn_circle["pop_name"].map(LGN_SHIFT) * rf_shift_vector * shift
    )

    # next, elongate the LGN complex orthogonal to the shift vector
    # rotate by shift vector to adjust the angle, strech, and rotate back.
    # sq_asr = np.sqrt(1.15)  # sqrt of aspect ratio (take from data later)
    sq_asr = np.sqrt(1.5)  # sqrt of aspect ratio (take from data later)
    lgn_relative = lgn_complex - rf_center
    lgn_rotate = lgn_relative / rf_shift_vector
    lgn_strech = lgn_rotate.real * sq_asr + 1j * lgn_rotate.imag / sq_asr
    lgn_back = lgn_strech * rf_shift_vector

    relative_rf_dist = np.abs(lgn_back)

    gauss_radius = 5.0  # extention of LGN axons in degrees
    gaussian_prob = gaussian_probability(relative_rf_dist, gauss_radius)

    # assign relative probability calculated above
    subunit_prob = np.zeros_like(gaussian_prob)
    subunit_dict_keys = []
    subunit_dict_values = []
    for subname, freqs in subunit_freqs.items():
        if cell_ignore_unit and (cell_ignore_unit in subname):
            # ignore either sON or sOFF
            continue

        probs = calculate_subunit_probs(cell_TF, [float(f) for f in freqs])
        for i in range(len(probs)):
            # construct the name
            typename = f"{subname}_TF{freqs[i]}"
            subunit_dict_keys.append(typename)
            subunit_dict_values.append(probs[i])

    subunit_dict = dict(zip(subunit_dict_keys, subunit_dict_values))
    subunit_prob = np.array(lgn_circle["pop_name"].map(subunit_dict).fillna(0.0))

    # treatments for sONsOFF and sONtOFF cells
    # tuning_angles are defined only for sONsOFF and sONtOFF cells, so this should be fine.
    lgn_ori_eligible = (
        delta_ori(lgn_circle["tuning_angle"] - target["tuning_angle"]) < 15
    )
    subunit_prob[lgn_ori_eligible] = 1.0

    total_prob = gaussian_prob * subunit_prob
    total_prob = total_prob / sum(total_prob)  # normalize
    if np.isnan(np.sum(total_prob)):
        print(f"NaN error {relative_rf_dist}; {gaussian_prob}; {subunit_prob}")

    # fraction of LGN synapses in e4's synapses. fixed parameter for this model
    e4_lgn_fraction = 0.2
    num_syns_orig = (
        target["target_sizes"]
        * e4_lgn_fraction
        * LGN_PARAMS.loc[pop_name, "synapse_ratio_against_e4"]
    )

    # We know the expected value for the number of synapses.
    # We estiamte the number of connections using it and the Yule distribution
    # parameter.
    yule_param = LGN_PARAMS.loc[pop_name, "yule_parameter"]
    num_cons = int(np.round(num_syns_orig * (yule_param - 1) / yule_param))

    # This line is to avoid crashing when you don't have sufficinet number of source
    # LGN neurons
    num_cons = min(num_cons, sum(total_prob > 0))
    selected_locs = pick_from_probs(num_cons, total_prob)

    # Now the source neurons are selected. Next set the number of synapses
    # draw from the Yule distribution (scipy)
    num_syns_indv = yulesimon.rvs(yule_param, size=num_cons)
    num_syns, num_neurons = np.unique(num_syns_indv, return_counts=True)
    selected_lgn_inds = lgn_circle.index[selected_locs]
    selected_lgn_dist = relative_rf_dist[selected_locs]

    nsyns_ret = np.zeros(len(lgn_nodes), dtype=int)
    for i in range(len(num_syns) - 1, 0, -1):
        gaussian_prob = gaussian_probability(
            selected_lgn_dist, gauss_radius / np.sqrt(num_syns[i])
        )
        gaussian_prob = gaussian_prob / sum(gaussian_prob)
        syn_selected = pick_from_probs(num_neurons[i], gaussian_prob)
        nsyns_ret[selected_lgn_inds[syn_selected]] = num_syns[i]

        selected_lgn_inds = np.delete(selected_lgn_inds, syn_selected)
        selected_lgn_dist = np.delete(selected_lgn_dist, syn_selected)

    # there should be 1 synapse connections remaining
    nsyns_ret[selected_lgn_inds] = 1

    return nsyns_ret
