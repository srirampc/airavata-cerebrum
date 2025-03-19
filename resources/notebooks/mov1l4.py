import marimo

__generated_with = "0.11.21"
app = marimo.App(width="full")


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
    main_rcp_file = "./v1l4/recipe/recipe.json"

    # recipe_dict = cbm_utils.io.load_json(main_rcp_file)
    # cmod_dict = cbm_utils.io.load_json(custom_mod_file)
    mdr_setup = cbm_setup.init_model_setup(
        name=m_name,
        model_base_dir=m_base_dir,
        recipe_files=rcp_files,
        recipe_dir=rcp_dir,
    )

    tree_view_widths = [0.4, 0.6]
    cmod_struct = cbm_structure.Network.from_file(custom_mod_file)

    integ_explorer = cbm_motree.RecipeExplorer(mdr_setup, cmod_struct).build("V1L4")
    integ_tree, integ_panels = integ_explorer.view_components()
    integ_motree = mo.ui.anywidget(integ_tree)
    return (
        awitree,
        cbm_motree,
        cbm_recipe,
        cbm_setup,
        cbm_structure,
        cbm_utils,
        cmod_struct,
        custom_mod_file,
        integ_explorer,
        integ_motree,
        integ_panels,
        integ_tree,
        json,
        m_base_dir,
        m_name,
        main_rcp_file,
        mdr_setup,
        mo,
        rcp_dir,
        rcp_files,
        tree_view_widths,
    )


@app.cell(hide_code=True)
def _(cbm_motree, integ_motree, integ_panels, mo, tree_view_widths):
    integ_selected = cbm_motree.TreeBase.panel_selector(integ_motree, integ_panels)
    mo.hstack(
       [
           integ_motree,
           integ_selected.layout if integ_selected else None
       ],
       widths=tree_view_widths
    )
    return (integ_selected,)


if __name__ == "__main__":
    app.run()
