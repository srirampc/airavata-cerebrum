import typing as t


#
# Lookup Keys in Configuration dictionary
@t.final
class RecipeKeys:
    RECIPE = "recipe"
    # Recipe Sections
    POST_OPS = "post_ops"
    DB2MODEL_MAP = "data2model_map"
    SRC_DATA = "source_data"
    TEMPLATES = "templates"
    #
    NODE_KEY = "node_key"
    NETWORK_STRUCT = "network_structure"
    NETWORK = "network"
    #
    LOCATIONS = "locations"
    CONNECTIONS = "connections"
    INIT_PARAMS = "init_params"
    EXEC_PARAMS = "exec_params"
    LABEL = "label"
    NAME = "name"
    TYPE = "type"
    WORKFLOW = "workflow"
    DB_CONNECT = "db_connect"
    #
    RECIPE_SECTIONS = [DB2MODEL_MAP, SRC_DATA]


#
# Lookup Keys in Configuration dictionary
@t.final
class RecipeLabels:
    INIT_PARAMS = " Init Arguments : "
    EXEC_PARAMS = " Exec Arguments : "
    NA = " N/A "


# File paths
@t.final
class RecipePaths:
    RECIPE_IO_DIR = "recipe"
