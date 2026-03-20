import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="SONATA Edge Editor Demo")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 🧠 SONATA Edge Editor — Interactive Demo

    This notebook demonstrates the `SonataEdgeEditor` context manager.

    **Workflow:**
    1. Upload a SONATA edge HDF5 file
    2. Configure the edge-type index (if present)
    3. Select an edge type to edit
    4. Choose what fraction of edges to randomly replace
    5. Run the replacement and compare before/after histograms
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    import altair as alt
    import tempfile
    import os
    import h5py
    from pathlib import Path
    from airavata_cerebrum.sonata.edge import SonataEdgeEditor, EditConfig, IndexType

    return (
        EditConfig,
        IndexType,
        Path,
        SonataEdgeEditor,
        alt,
        h5py,
        os,
        pl,
        tempfile,
    )


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 1 — Select a SONATA edge file
    """)
    return


@app.cell
def _(mo, os):
    file_browser = mo.ui.file_browser(
        initial_path=os.getcwd(),
        filetypes=[".h5", ".hdf5"],
        label="Browse to a SONATA edge HDF5 file (leave unselected to use built-in demo file)",
        multiple=False,
    )
    file_browser
    return (file_browser,)


@app.cell
def _(Path, file_browser, mo):
    if file_browser.value:
        h5_path = Path(file_browser.value[0].path)
        _source_label = f"📂 Selected: `{h5_path}`"
    else:
        h5_path = None

    mo.callout(mo.md(_source_label), kind="info")
    return (h5_path,)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 2 — Inspect file
    """)
    return


@app.cell
def _(EditConfig, IndexType, SonataEdgeEditor, h5_path, mo):
    et_cfg = EditConfig(
            edge_type_index_group = "edge_type_to_index",
            type_to_range_dataset = "node_id_to_range",
            range_to_edge_dataset = "range_to_edge_id",
    )

    with SonataEdgeEditor(h5_path, edge_type_index_cfg=et_cfg) as _ed:
        _idx   = _ed.has_indices()
        _pop   = _ed.population

        # read full edge table to get type counts
        import h5py as _h5py
        with _h5py.File(h5_path, "r") as _f:
            _grp = _f[f"edges/{_pop}"]
            import numpy as _np
            _etypes = _grp["edge_type_id"][:]

    import polars as _pl
    counts_df = (
        _pl.DataFrame({"edge_type_id": _etypes})
        .group_by("edge_type_id")
        .agg(_pl.len().alias("count"))
        .sort("edge_type_id")
    )
    _type_ids = counts_df["edge_type_id"].to_list()
    # expose for downstream cells
    file_population = _pop
    file_type_ids   = _type_ids


    mo.vstack([
        mo.stat(label="Population",       value=_pop,                      bordered=True),
        mo.stat(label="Total edges",       value=str(len(_etypes)),         bordered=True),
        mo.stat(label="Unique edge types", value=str(len(_type_ids)),       bordered=True),
        mo.stat(label="Node index",        value="✅ present" if IndexType.NODE in _idx else "❌ absent", bordered=True),
        mo.stat(label="Edge-type index",   value="✅ present" if IndexType.EDGE_TYPE in _idx else "❌ absent", bordered=True),
    ], justify="start", gap="1rem")
    return counts_df, et_cfg, file_type_ids


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 3 — Edge-type distribution *before* editing
    """)
    return


@app.cell
def _(alt, counts_df):
    chart_before = (
        alt.Chart(counts_df.to_pandas().head(15), title="Edge counts by type — BEFORE")
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("edge_type_id:O", title="Edge type ID", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
               "count:Q",
                title="Number of edges", 
                scale=alt.Scale(type="linear"),
                axis=alt.Axis(format="~s")
            ),
            color=alt.Color(
                    "edge_type_id:O",
                    scale=alt.Scale(scheme="tableau10"),
                    legend=None,
            ),
            tooltip=["edge_type_id:O", "count:Q"],
        )
        .properties(width=420, height=280)
    )
    chart_before
    return (chart_before,)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 5 — Configure the replacement
    """)
    return


@app.cell
def _(file_type_ids, mo):
    edge_type_selector = mo.ui.dropdown(
        options={str(t): t for t in file_type_ids},
        label="Edge type to edit",
        value=str(file_type_ids[0]),
    )
    replace_fraction = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.5,
        label="Fraction of edges to randomly replace",
        show_value=True,
    )
    mo.vstack([edge_type_selector, replace_fraction])
    return edge_type_selector, replace_fraction


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="▶ Run replacement")
    run_button
    return (run_button,)


@app.cell
def _(
    Path,
    SonataEdgeEditor,
    edge_type_selector,
    et_cfg,
    h5_path,
    mo,
    replace_fraction,
    run_button,
    tempfile,
):
    mo.stop(not run_button.value, mo.callout(
        mo.md("👆 Press **Run replacement** to execute."), kind="neutral"
    ))

    _etype      = int(edge_type_selector.value)
    _frac       = float(replace_fraction.value)

    # write output to a temp file so we never modify the source
    _out_tmp    = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    _out_tmp.close()
    out_h5_path = Path(_out_tmp.name)

    with SonataEdgeEditor(h5_path, edge_type_index_cfg=et_cfg) as _ed:
        _edges = _ed.get_edges_by_type(_etype)
        _n_total = len(_edges)

        # sample a random subset using polars — fraction argument avoids manual n calculation
        _replacement = _edges.sample(fraction=_frac, shuffle=True)

        _ed.replace_edges_by_type(
            edge_type_id=_etype,
            new_edges=_replacement,
            output_path=out_h5_path,
        )

    _summary = (
        f"**Edge type `{_etype}`:** original {_n_total} edges → "
        f"replaced with a random subset of **{len(_replacement)}** edges "
        f"({int(_frac * 100)}% of the original set)."
    )
    mo.callout(mo.md(_summary), kind="success")
    return (out_h5_path,)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 6 — Edge-type distribution *after* editing
    """)
    return


@app.cell
def _(alt, chart_before, h5py, mo, out_h5_path, pl):
    # read type counts from output file
    with h5py.File(out_h5_path, "r") as _f:
        _pop_after = list(_f["edges"].keys())[0]
        _etypes_after = _f[f"edges/{_pop_after}/edge_type_id"][:]

    counts_after = (
        pl.DataFrame({"edge_type_id": _etypes_after})
        .group_by("edge_type_id")
        .agg(pl.len().alias("count"))
        .sort("edge_type_id")
    )

    chart_after = (
        alt.Chart(counts_after.to_pandas().head(15), title="Edge counts by type — AFTER")
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("edge_type_id:O", title="Edge type ID", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "count:Q", title="Number of edges",
                #scale=alt.Scale(type="log", base=10, domainMin=1),
                axis=alt.Axis(format="~s")
            ),
            color=alt.Color(
                "edge_type_id:O",
                scale=alt.Scale(scheme="tableau10"),
                legend=None,
            ),
            tooltip=["edge_type_id:O", "count:Q"],
        )
        .properties(width=420, height=280)
    )
    mo.hstack([chart_before, chart_after], align="center")
    return (counts_after,)


@app.cell
def _(counts_after, counts_df, mo, pl):
    _diff = (
        counts_df.rename({"count": "before"})
        .join(counts_after.rename({"count": "after"}), on="edge_type_id", how="full")
        .sort("edge_type_id")
        .with_columns(
            (pl.col("before") - pl.col("after")).alias("delta")
        )
    )
    mo.vstack([
        mo.md("### Before / After comparison"),
        mo.ui.table(_diff.to_pandas(), selection=None),
    ])
    return


@app.cell
def _(mo, out_h5_path):
    with open(out_h5_path, "rb") as _fh:
        _bytes = _fh.read()

    mo.vstack([
        mo.md("---\n## Download modified file"),
        mo.download(
            data=_bytes,
            filename="edges_modified.h5",
            mimetype="application/octet-stream",
            label="⬇ Download edges_modified.h5",
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
