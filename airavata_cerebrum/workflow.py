import logging
import typing as t
import duckdb
import tqdm.contrib.logging as tqdm_log

from collections.abc import Iterable
from . import register, base
from .model.setup import RecipeKeys

def _log():
    return logging.getLogger(__name__)


def run_workflow(
    workflow_steps: list[dict[str, t.Any]] | None,
    wf_iter: Iterable[dict[str, t.Any]] | None = None
) -> Iterable[dict[str, t.Any]] | None:
    if workflow_steps is None:
        return None
    for wf_step in workflow_steps:
        sname : str = wf_step[RecipeKeys.NAME]
        slabel : str = wf_step[RecipeKeys.LABEL] if RecipeKeys.LABEL in wf_step else sname
        iparams: dict[str, t.Any] = wf_step[RecipeKeys.INIT_PARAMS]
        eparams: dict[str, t.Any] = wf_step[RecipeKeys.EXEC_PARAMS]
        match wf_step[RecipeKeys.TYPE]:
            case "query":
                _log().info("Start Query : [%s]",  slabel)
                qobj: base.DbQuery | None = register.get_query_object(
                    sname, **iparams
                )
                if qobj:
                    wf_iter = qobj.run(wf_iter, **eparams)
                    _log().info("Complete Query : [%s]", slabel)
                else:
                    _log().error("Failed to find Query : [%s]",  sname)
            case "xform":
                _log().info("Running XFormer : [%s]",  slabel)
                fobj: base.OpXFormer | None = register.get_xform_op_object(
                    sname, **iparams
                )
                if fobj and wf_iter:
                    wf_iter = fobj.xform(wf_iter, **eparams)
                    _log().info("Complete XForm : [%s]", slabel)
                else:
                    _log().error("Failed to find XFormer : [%s]", sname)
            case _:
                pass
    return wf_iter


def run_db_connect_workflows(
    source_data_cfg: dict[str, t.Any]
) -> dict[str, t.Any]:
    db_connect_output = {}
    #
    for db_name, db_wcfg in source_data_cfg.items():
        db_label = db_name
        if RecipeKeys.LABEL in db_wcfg:
            db_label = db_wcfg[RecipeKeys.LABEL]
        _log().info("Start db_connect workflow for db: [%s]",  db_label)
        with tqdm_log.logging_redirect_tqdm():
            model_itr = run_workflow(
                db_wcfg[RecipeKeys.DB_CONNECT][RecipeKeys.WORKFLOW]
            )
            if model_itr:
                db_connect_output[db_name] = list(model_itr)
        _log().info("Complete db_connect workflow for db: [%s]", db_label)
    #
    return db_connect_output


def run_ops_workflows(
    db_conn_data: dict[str, t.Any],
    ops_config_desc: dict[str, t.Any],
    ops_key: str | None = None
) -> dict[str, t.Any]:
    op_output_data = {}
    for src_db, op_config in ops_config_desc.items():
        _log().info("Start op workflow for db [%s]", src_db)
        wf_input = db_conn_data[src_db]
        op_desc = op_config[ops_key] if ops_key else op_config
        op_output = run_workflow(
            op_desc[RecipeKeys.WORKFLOW],
            wf_input
        )
        op_output = list(op_output) if op_output else []
        _log().debug(
            "WF Desc: [%s]; WF IN: [%s]; Op output: [%s]",
            str(op_desc),
            str(len(wf_input)),
            str(len(op_output))
        )
        op_output_data[src_db] = op_output
        _log().info("Complete op workflow for db [%s]", src_db)
    return op_output_data


def map_srcdata_locations(
    source_data: dict[str, t.Any],
    data2loc_map: dict[str, t.Any],
) -> dict[str, t.Any]:
    net_locations = {}
    for location, location_desc in data2loc_map.items():
        neuron_desc_map = {}
        for neuron, neuron_dcfg in location_desc.items():
            _log().info("Processing db connection for neuron [%s]", neuron)
            ops_dict = neuron_dcfg[RecipeKeys.SRC_DATA]
            neuron_dc_map = run_ops_workflows(source_data, ops_dict)
            for dkey in neuron_dcfg.keys():
                if dkey != RecipeKeys.SRC_DATA:
                    neuron_dc_map[dkey] = neuron_dcfg[dkey]
            neuron_desc_map[neuron] = neuron_dc_map
            _log().info("Completed db connection for neuron [%s]", neuron)
        net_locations[location] = neuron_desc_map
    return net_locations

def map_srcdata_connections(
    source_data: dict[str, t.Any],
    data2con_map: dict[str, t.Any],
) -> dict[str, t.Any]:
    net_connections = {}
    for connx, connx_desc in data2con_map.items():
        conn_desc_map = {}
        _log().info("Processing db data for connex [%s]", connx)
        ops_dict = connx_desc[RecipeKeys.SRC_DATA]
        conn_desc_map = run_ops_workflows(source_data, ops_dict)
        for dkey in connx_desc.keys():
            if dkey != RecipeKeys.SRC_DATA:
                conn_desc_map[dkey] = connx_desc[dkey]
        _log().info("Completed db data for connex [%s]", connx)
        net_connections[connx] = conn_desc_map
    return net_connections

def write_db_connect_duck(
    db_connect_output: dict[str, t.Any],
    db_name: str,
) -> None:
    #
    with duckdb.connect(db_name) as db_conn:
        for db_name, db_wout in db_connect_output.items():
            db_writer = register.get_query_db_writer_object(
                db_name,
                db_conn=db_conn
            )
            db_writer.write(db_wout)
