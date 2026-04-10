import h5py
import pandas as pd

def spikes_dataf(spikes_h5: str):
    with h5py.File(spikes_h5) as sfx:
        sfspikes = sfx['/spikes/v1']
        snodes = sfspikes['node_ids']
        stimes = sfspikes['timestamps']
        return pd.DataFrame(data={
            'node_id':snodes[:snodes.len()],
            'timestamp':stimes[:stimes.len()]
        })
    return None


def nodes_dataf(nodes_h5: str):
    with h5py.File(nodes_h5) as vifx:
        ntypes = vifx['/nodes/v1/node_type_id']
        nyes = vifx['/nodes/v1/0/y']
        nids = vifx['/nodes/v1/node_id']
        return pd.DataFrame(data={
            'node_id':nids[:nids.len()],
            'y':nyes[:nyes.len()],
            'node_type_id': ntypes[:ntypes.len()]
        })

def node_types_dataf(nodes_types_csv: str):
    ntypesdf = pd.read_csv(nodes_types_csv, sep=' ')
    ntypesdf['layer'] = ntypesdf['pop_name'].str.replace('(', '', regex=False).str.replace(')', '', regex=False).str.split(',').str[0]
    ntypesdf['cell_type'] = ntypesdf['pop_name'].str.replace('(', '', regex=False).str.replace(')', '', regex=False).str.split(',').str[-1]
    return ntypesdf[['node_type_id', 'ei', 'location','pop_name', 'layer', 'cell_type']]



