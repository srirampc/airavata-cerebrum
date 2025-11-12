from airavata_cerebrum.model.setup import RecipeSetup
import airavata_cerebrum.view.tree as cbm_tree

mdr_setup = RecipeSetup(
    name="v1l4",
    base_dir="./",
    config_files={"config": ["config.json"],
                  "templates": ["config_template.json"]},
    config_dir="./v1l4/description/",
)
print("Checking if mandatory config keys exists : ", mdr_setup.valid())
print("Config loaded with ", len(mdr_setup.get_templates()), " templates")

# TODO:
src_tree = cbm_tree.SourceDataTreeView(mdr_setup)
src_tree.build()

loc_tree = cbm_tree.D2MLocationsTreeView(mdr_setup)
loc_tree.build()

conn_tree = cbm_tree.D2MConnectionsTreeView(mdr_setup)
conn_tree.build()
