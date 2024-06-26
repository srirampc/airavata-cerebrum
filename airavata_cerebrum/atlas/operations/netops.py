import pandas as pd
import numpy as np
import scipy
import scipy.stats
import typing
from ..data import abc_mouse
from ..model import regions


def atlasdata2regionfractions(
    region_frac_df: pd.DataFrame, model_name: str
) -> regions.Network:
    loc_struct = {}
    for loc, row in region_frac_df.iterrows():
        neuron_struct = {}
        for gx in abc_mouse.GABA_TYPES:
            frac_col = abc_mouse.FRACTION_COLUMN_FMT.format(gx)
            neuron_struct[gx] = regions.Neuron(ei="i", fraction=float(row[frac_col]))
        for gx in abc_mouse.GLUT_TYPES:
            frac_col = abc_mouse.FRACTION_COLUMN_FMT.format(gx)
            neuron_struct[gx] = regions.Neuron(ei="e", fraction=float(row[frac_col]))
        loc_struct[loc] = regions.Region(
            name=str(loc),
            inh_fraction=float(row[abc_mouse.INHIBITORY_FRACTION_COLUMN]),
            region_fraction=float(row[abc_mouse.FRACTION_WI_REGION_COLUMN]),
            neurons=neuron_struct,
        )
    return regions.Network(name=model_name, locations=loc_struct)


def subset_network(net_stats: regions.Network, 
                   region_list: typing.List[str]) -> regions.Network:
    sub_locs = {k:v  for k,v in net_stats.locations.items() if k in region_list}
    return regions.Network(name=net_stats.name, dims=net_stats.dims,
                           locations=sub_locs)


def update_user_input(
    net_stats: regions.Network, upd_stats: regions.Network
) -> regions.Network:
    # update dims
    for upkx, upvx in upd_stats.dims.items():
        net_stats.dims[upkx] = upvx
    upd_locations = upd_stats.locations
    net_locations = net_stats.locations
    # Locations
    for lx, uprx in upd_locations.items():
        if lx not in net_locations:
            net_stats.locations[lx] = uprx
            continue
        # update fractions
        if uprx.inh_fraction > 0:
            net_stats.locations[lx].inh_fraction = uprx.inh_fraction
        if uprx.region_fraction > 0:
            net_stats.locations[lx].region_fraction = uprx.region_fraction
        # update ncells
        if uprx.ncells > 0:
            net_stats.locations[lx].ncells = uprx.ncells
        if uprx.inh_ncells > 0:
            net_stats.locations[lx].inh_ncells = uprx.inh_ncells
        if uprx.exc_ncells > 0:
            net_stats.locations[lx].exc_ncells = uprx.exc_ncells
        # update dimensions
        for upkx, upvx in uprx.dims.items():
            net_stats.locations[lx].dims[upkx] = upvx
        # update neuron details
        for nx, sx in uprx.neurons.items():
            if nx not in net_stats.locations[lx].neurons:
                net_stats.locations[lx].neurons[nx] = sx
                continue
            if sx.fraction > 0:
                net_stats.locations[lx].neurons[nx].fraction = sx.fraction
            if sx.N > 0:
                net_stats.locations[lx].neurons[nx].N = sx.N
            # update dimensions
            for upkx, upvx in uprx.neurons[nx].dims.items():
                net_stats.locations[lx].neurons[nx].dims[upkx] = upvx
            # update model items
            # model_name : str | None = None
            # model_type: str | None =  None
            # model_template: str | None = None
            # dynamics_params: str | None = None 
            if sx.model_name is not None:
                net_stats.locations[lx].neurons[nx].model_name = sx.model_name
            if sx.model_type is not None:
                net_stats.locations[lx].neurons[nx].model_type = sx.model_type
            if sx.model_template is not None:
                net_stats.locations[lx].neurons[nx].model_template = sx.model_template
            if sx.dynamics_params is not None:
                net_stats.locations[lx].neurons[nx].model_template = sx.dynamics_params
    return net_stats


def fractions2ncells(net_stats: regions.Network, N: int) -> regions.Network:
    net_stats.ncells = N
    for lx, lrx in net_stats.locations.items():
        ncells_region = int(lrx.region_fraction * N)
        ncells_inh = int(lrx.inh_fraction * ncells_region)
        ncells_exc = ncells_region - ncells_inh
        net_stats.locations[lx].ncells = ncells_region
        net_stats.locations[lx].inh_ncells = ncells_inh
        net_stats.locations[lx].exc_ncells = ncells_exc
        for nx, nurx in lrx.neurons.items():
            eix = nurx.ei
            ncells = ncells_inh if eix == "i" else ncells_exc
            ncells = int(ncells * nurx.fraction)
            if ncells == 0:
                continue
            net_stats.locations[lx].neurons[nx].N = ncells
    return net_stats


def generate_random_cyl_pos(N, layer_range, radial_range):
    radius_outer = radial_range[1]
    radius_inner = radial_range[0]

    phi = 2.0 * np.pi * np.random.random([N])
    r = np.sqrt(
        (radius_outer**2 - radius_inner**2) * np.random.random([N])
        + radius_inner**2
    )
    x = r * np.cos(phi)
    z = r * np.sin(phi)

    layer_start = layer_range[0]
    layer_end = layer_range[1]
    # Generate N random z values.
    y = (layer_end - layer_start) * np.random.random([N]) + layer_start

    positions = np.column_stack((x, y, z))

    return positions


def generate_target_sizes(N, ln_shape, ln_scale):
    ln_rv = scipy.stats.lognorm(s=ln_shape, loc=0, scale=ln_scale)
    ln_rvs = ln_rv.rvs(N).round()
    return ln_rvs


def generate_node_positions(model_struct):
    pass


def map_node_paramas(model_struct, node_map):
    pass


def filter_node_params(model_struct, filter_predicate):
    pass


def map_edge_paramas(model_struct, node_map):
    pass


def filter_edge_params(model_struct, filter_predicate):
    pass


def union_network(model_struct1, model_struct2):
    pass


def join_network(model_struct1, model_struct2):
    pass
