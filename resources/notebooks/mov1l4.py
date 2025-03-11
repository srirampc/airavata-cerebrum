import marimo

__generated_with = "0.11.17"
app = marimo.App()


@app.cell
def _():
    import json
    import airavata_cerebrum.model.setup as cbm_setup
    import airavata_cerebrum.model.recipe as cbm_recipe
    import airavata_cerebrum.view.motree as cbm_motree
    import airavata_cerebrum.model.structure as cbm_structure
    import airavata_cerebrum.util as cbm_utils
    import marimo as mo
    import awitree

    custom_mod_file = "./v1l4/recipe/custom_mod.json"
    m_base_dir = "./"
    m_name = "v1l4"
    rcp_files = {"recipe": ["recipe.json"], "templates": ["recipe_template.json"]}
    rcp_dir = "./v1l4/recipe/"

    recipe_dict = cbm_utils.io.load_json("./v1l4/recipe/recipe.json")
    cmod_dict = cbm_utils.io.load_json(custom_mod_file)
    mdr_setup = cbm_setup.init_model_setup(
        name=m_name,
        model_base_dir=m_base_dir,
        recipe_files=rcp_files,
        recipe_dir=rcp_dir,
    )

    tree_view_widths = [0.4, 0.6]
    cmod_struct = cbm_structure.Network.model_validate(cmod_dict)

    stree, spanel_dict = cbm_motree.SourceDataTreeView(mdr_setup).build().view_comps()
    smotree = mo.ui.anywidget(stree)
    return (
        awitree,
        cbm_motree,
        cbm_recipe,
        cbm_setup,
        cbm_structure,
        cbm_utils,
        cmod_dict,
        cmod_struct,
        custom_mod_file,
        json,
        m_base_dir,
        m_name,
        mdr_setup,
        mo,
        rcp_dir,
        rcp_files,
        recipe_dict,
        smotree,
        spanel_dict,
        stree,
        tree_view_widths,
    )


@app.cell(hide_code=True)
def _(cbm_motree, mo, smotree, spanel_dict, stree, tree_view_widths):
    srt_selected = cbm_motree.TreeBase.panel_selector(stree, spanel_dict)
    mo.hstack(
        [
            smotree,
            srt_selected.layout if srt_selected else None
        ],
        widths=tree_view_widths
    )
    return (srt_selected,)


@app.cell
def _(cbm_motree, mdr_setup, mo):
    d2mltree, d2ml_panel_dict = cbm_motree.D2MLocationsTreeView(mdr_setup).build().view_comps()
    d2ml_motree = mo.ui.anywidget(d2mltree)
    return d2ml_motree, d2ml_panel_dict, d2mltree


@app.cell(hide_code=True)
def _(
    cbm_motree,
    d2ml_motree,
    d2ml_panel_dict,
    d2mltree,
    mo,
    tree_view_widths,
):
    d2mlrt_selected = cbm_motree.TreeBase.panel_selector(d2mltree, d2ml_panel_dict)
    mo.hstack(
        [
            d2ml_motree,
            d2mlrt_selected.layout if d2mlrt_selected else mo.vstack([])
        ],
        widths=tree_view_widths
    )
    return (d2mlrt_selected,)


@app.cell
def _(cbm_motree, mdr_setup, mo):
    d2mctree, d2mc_panel_map = cbm_motree.D2MConnectionsTreeView(mdr_setup).build().view_comps()
    d2mc_motree = mo.ui.anywidget(d2mctree)
    return d2mc_motree, d2mc_panel_map, d2mctree


@app.cell(hide_code=True)
def _(cbm_motree, d2mc_motree, d2mc_panel_map, mo, tree_view_widths):
    d2mcrt_selected = cbm_motree.TreeBase.panel_selector(d2mc_motree, d2mc_panel_map)
    mo.hstack(
        [
            d2mc_motree,
            d2mcrt_selected.layout if d2mcrt_selected else None
        ],
        widths=tree_view_widths
    )
    return (d2mcrt_selected,)


@app.cell
def _(cbm_motree, cmod_struct, mo):
    cmod_tview = cbm_motree.NetworkTreeView(cmod_struct).build()
    cmodtree, cmod_panel_dict = cmod_tview.view_comps()
    cmod_motree = mo.ui.anywidget(cmodtree)
    return cmod_motree, cmod_panel_dict, cmod_tview, cmodtree


@app.cell
def _(cbm_motree, cmod_motree, cmod_panel_dict, mo, tree_view_widths):
    cmod_selected = cbm_motree.TreeBase.panel_selector(cmod_motree, cmod_panel_dict)
    mo.hstack(
        [
            cmod_motree,
            cmod_selected.layout if cmod_selected else None
        ],
        widths=tree_view_widths
    )
    return (cmod_selected,)


if __name__ == "__main__":
    app.run()
