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
def _(recipe_dict):
    recipe_dict
    return


@app.cell
def _(mdr_setup):
    mdr_setup.recipe_sections["templates"]
    cpfilter = mdr_setup.get_template_for("airavata_cerebrum.operations.abm_celltypes.CTPropertyFilter")
    cpfilter
    return (cpfilter,)


@app.cell
def _(cbm_motree, cpfilter):

    spanel = cbm_motree.RecipeSidePanel(cpfilter)
    spanel.layout
    return (spanel,)


@app.cell
def _(mo):
    mo.state
    return


if __name__ == "__main__":
    app.run()
