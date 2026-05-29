import h5py
import pandas as pd


def spikes_df(spikes_h5):
    with h5py.File(spikes_h5) as sfx:
        sfspikes = sfx['/spikes/v1']
        snodes = sfspikes['node_ids']  # pyright: ignore[reportIndexIssue]
        stimes = sfspikes['timestamps']  # pyright: ignore[reportIndexIssue]
        return pd.DataFrame(data={
            'node_id': snodes[:snodes.len()],  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
            'timestamp': stimes[:stimes.len()]  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        })


def nodes_df(nodes_h5):
    with h5py.File(nodes_h5) as vifx:
        ntypes = vifx['/nodes/v1/node_type_id']
        nyes = vifx['/nodes/v1/0/y']
        nids = vifx['/nodes/v1/node_id']
        return pd.DataFrame(data={
            'node_id':nids[:nids.len()],  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
            'y':nyes[:nyes.len()],  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
            'node_type_id': ntypes[:ntypes.len()],  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        })


def lgn_df(lgn_h5):
    with h5py.File(lgn_h5) as lfx:
        ltimes = lfx['spikes/timestamps']
        lgids = lfx['spikes/gids']
        return pd.DataFrame(data={
            'gid': lgids[:lgids.len()],  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
            'time': ltimes[:ltimes.len()],   # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        })


NODE_TYPES_COLUMNS = [
    'node_type_id', 'ei', 'location','pop_name', 'layer', 'cell_type'
]


def node_types_df(nodes_types_csv):
    ntypesdf = pd.read_csv(nodes_types_csv, sep=' ')
    ntypesdf['layer'] = ntypesdf['pop_name'].str.replace(
        '(', '', regex=False
    ).str.replace(')', '', regex=False).str.split(',').str[0]
    ntypesdf['cell_type'] = ntypesdf['pop_name'].str.replace(
        '(', '', regex=False
    ).str.replace(')', '', regex=False).str.split(',').str[-1]
    return ntypesdf[NODE_TYPES_COLUMNS]
