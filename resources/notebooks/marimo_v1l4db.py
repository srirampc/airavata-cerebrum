import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import ast
    import itertools
    import marimo as mo
    import airavata_cerebrum.util.io as cuio
    import airavata_cerebrum.dataset.abc_mouse as abcm
    import airavata_cerebrum.dataset.abm_celltypes as abmct
    import airavata_cerebrum.dataset.ai_synphys as aisyn
    import airavata_cerebrum.workflow as abmwf
    import airavata_cerebrum.register as abmreg
    import duckdb
    import polars as pl
    import typing as t

    src_data_file = "./v1l4/recipe/db_connect_output.json"
    test_db = "./v1l4/recipe/test.db"
    src_data = cuio.load_json(src_data_file)
    test_conn = duckdb.connect(test_db)
    #db_conn = duckdb.connect("./v1l4/recipe/db_connect_output.db")
    return (
        abcm,
        abmct,
        abmreg,
        abmwf,
        aisyn,
        ast,
        cuio,
        duckdb,
        itertools,
        mo,
        pl,
        src_data,
        src_data_file,
        t,
        test_conn,
        test_db,
    )


@app.cell
def _(abmwf, src_data):
    abmwf.write_db_connect_duck(
        src_data,
        "./v1l4/recipe/db_connect_output.db"
    )
    return


@app.cell
def _(abmct, src_data):
    tlist = src_data['airavata_cerebrum.dataset.abm_celltypes']
    # dfx = pl.DataFrame((x['ct'] | (x['glif'] if x['glif'] else {}) for x in tlist))
    abmct.DFBuilder.build(tlist, df_source="glif")
    return (tlist,)


@app.cell
def _(abmct, tlist):
    abmct.DFBuilder.build(tlist, df_source="nm")
    return


@app.cell
def _():
    #abmct.DuckDBWriter(test_conn).write(tlist)
    return


@app.cell
def _(aisyn, src_data):
    slist = src_data["airavata_cerebrum.dataset.ai_synphys"]
    aisyn.DFBuilder.build(slist)
    return (slist,)


@app.cell
def _(abcm, src_data):
    alist = src_data["airavata_cerebrum.dataset.abc_mouse"]
    abcm.DFBuilder.build(alist)
    return (alist,)


if __name__ == "__main__":
    app.run()
