import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import airavata_cerebrum.model.setup as cbm_setup
    import airavata_cerebrum.model.recipe as cbm_recipe
    import airavata_cerebrum.view.motree as cbm_motree
    import marimo as mo
    import awitree

    custom_mod_file = "./v1l4/recipe/custom_mod.json"
    m_base_dir = "./"
    m_name = "v1l4"
    rcp_files = {"recipe": ["recipe.json"], "templates": ["recipe_template.json"]}
    rcp_dir = "./v1l4/recipe/"

    recipe_dict = {}
    with open("./v1l4/recipe/recipe.json") as ifx:
        recipe_dict = json.load(ifx)

    custom_mod_dict = {}
    with open(custom_mod_file) as ifx:
        custom_mod_dict = json.load(ifx)

    mdr_setup = cbm_setup.init_model_setup(
        name=m_name,
        model_base_dir=m_base_dir,
        recipe_files=rcp_files,
        recipe_dir=rcp_dir,
    )
    return (
        awitree,
        cbm_motree,
        cbm_recipe,
        cbm_setup,
        custom_mod_dict,
        custom_mod_file,
        ifx,
        json,
        m_base_dir,
        m_name,
        mdr_setup,
        mo,
        rcp_dir,
        rcp_files,
        recipe_dict,
    )


@app.cell
def _(cbm_motree, mdr_setup, mo):
    src_tree = cbm_motree.SourceDataTreeView(mdr_setup).build()
    stree = src_tree.tree
    spanel_dict = src_tree.panel_dict
    smotree = mo.ui.anywidget(stree)
    return smotree, spanel_dict, src_tree, stree


@app.cell
def _(mo, smotree, spanel_dict, src_tree, stree):
    srt_selected = (
        spanel_dict[stree.selected_nodes[0]["id"]]
        if (
            stree.selected_nodes and
            len(stree.selected_nodes) > 0 and
            (stree.selected_nodes[0]["id"] in spanel_dict)
        ) else None
    )
    mo.hstack(
        [
            smotree,
            srt_selected.layout if srt_selected else None
        ],
        widths=src_tree.widths
    )
    return (srt_selected,)


@app.cell
def _(rtree):
    rtree.selected_nodes
    return


@app.cell
def _(awitree):
    rtree = awitree.Tree(data={
            "id": "0",
            "text":"Main Root",
            "state": {"open" : True},
            "children" : [
                {"id": "1", "text" : "Sub Node 1", "children":[]},
                {"id": "2", "text" : "Sub Node 2", "children":[]},
                {"id": "3", "text" : "Sub Node 3", "children":[]},
            ]
    })
    rtree
    return (rtree,)


if __name__ == "__main__":
    app.run()
