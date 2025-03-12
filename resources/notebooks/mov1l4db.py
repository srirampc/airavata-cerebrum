import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import airavata_cerebrum.util.io as cuio
    import duckdb
    import polars as pl

    src_data_file = "./v1l4/recipe/db_connect_output.json"
    src_data = cuio.load_json(src_data_file)
    return cuio, duckdb, mo, pl, src_data, src_data_file


@app.cell
def _(src_data):
    src_data
    return


@app.cell
def _(pl, src_data):
    pl.DataFrame(
        list(src_data["airavata_cerebrum.dataset.ai_synphys"][0].items()),
        schema=[("Pair", pl.String),("Ratio", pl.Float64)],
        orient="row"
    )
    return


@app.cell
def _(src_data):
    alist = src_data["airavata_cerebrum.dataset.abc_mouse"]
    result = []
    for clist in alist:
        for px, bdct in clist.items():
            for rx,y in bdct.items():
                result.append(y)
    result
    return alist, bdct, clist, px, result, rx, y


@app.cell
def _(pl, result):
    pl.DataFrame(
    result
    )

    return


if __name__ == "__main__":
    app.run()
