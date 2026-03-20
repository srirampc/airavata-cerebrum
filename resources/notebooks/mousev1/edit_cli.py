from airavata_cerebrum.sonata.edge import IndexType, SonataEdgeEditor, EditConfig
import polars as pl

if __name__ == "__main__":
    H5_FILE   = "./model_prior/v1_test/network_v1/left_local_edges.edge_types_120.sorted.h5"
    OUT_H5_FILE   = "./model_prior/v1_test/network_v1/left_local_edges.edge_types_120.modified.h5"
    EDGE_TYPE = 117
    THRESHOLD =  0.7

    # Describe the non-standard edge-type index in your file.
    # Set to None (or omit) if your file does not have one.
    cfg = EditConfig(
        edge_type_index_group="edge_type_to_index",
        type_to_range_dataset="node_id_to_range",   # BBP naming (likely a typo)
        range_to_edge_dataset="range_to_edge_id",
    )

    with SonataEdgeEditor(H5_FILE, edge_type_index_cfg=cfg) as editor:
        print(editor)  # SonataEdgeEditor(path='edges.h5', population='...', state=open)

        # Check which indices exist
        status = editor.has_indices()
        if status:
            print(f"Node index      : {IndexType.NODE in status}")
            print(f"Edge-type index : {IndexType.EDGE_TYPE in status}")
        else:
            print("No Indices")

        # Fetch edges of a given type
        edges = editor.get_edges_by_type(EDGE_TYPE)
        print(f"\nFound {len(edges)} edges of type {EDGE_TYPE}")
        print(edges.head())

        # Modify and write — indices rebuilt automatically
        new_edges = edges.drop("_row_index").sample(fraction=THRESHOLD)
        print(f"\nSampled {len(new_edges)} edges of type {EDGE_TYPE}")
        print(new_edges.head())
        ##if "syn_weight" in new_edges.columns:
        ##    new_edges = new_edges.with_columns(pl.col("syn_weight") * 2.0)

        out = editor.replace_edges_by_type(
            edge_type_id=EDGE_TYPE,
            new_edges=new_edges,
            output_path=OUT_H5_FILE,   # omit to edit in-place
        )
        print(f"\nOutput written to: {out}")

    with SonataEdgeEditor(OUT_H5_FILE, edge_type_index_cfg=cfg) as editor:
        # Fetch edges of a given type
        edges = editor.get_edges_by_type(EDGE_TYPE)
        print(f"\nFound {len(edges)} edges of type {EDGE_TYPE}")
        print(edges.head())

        # Rebuild indices manually on any file if needed:
        # editor.rebuild_indices("edges_modified.h5", which="all")
