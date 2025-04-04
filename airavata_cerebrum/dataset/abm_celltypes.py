import itertools
import json
import logging
import os
#
import duckdb
import polars as pl
import typing as t
import allensdk.core.cell_types_cache
import allensdk.api.queries.cell_types_api
#
from typing_extensions import override
from pydantic import Field
from allensdk.api.queries.glif_api import GlifApi
from pathlib import Path
#
from ..base import (
    CerebrumBaseModel,
    DbQuery,
    BaseParams,
    NoneParams,
    QryDBWriter,
    QryItr,
)
from ..util import exclude_keys, prefix_keys


def _log():
    return logging.getLogger(__name__)


class CCacheInitParams(CerebrumBaseModel):
    download_base : t.Annotated[str, Field(title="Download Base Dir.")]

class CCacheExecParams(CerebrumBaseModel):
    species  : t.Annotated[str | None, Field(title="Species")] = None
    manifest : t.Annotated[str, Field(title="Manifest File")] = "manifest.json"
    cells    : t.Annotated[str, Field(title="Cells json File")]  = "cells.json"

CCacheBaseParams : t.TypeAlias = BaseParams[CCacheInitParams, CCacheExecParams] 

class CTDbCellCacheQuery(DbQuery[CCacheInitParams, CCacheExecParams]):
    class QryParams(CCacheBaseParams):
        init_params: t.Annotated[CCacheInitParams, Field(title='Init Params')]
        exec_params: t.Annotated[CCacheExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: CCacheInitParams, **params: t.Any):
        """
        Initialixe Cache Query
        Parameters
        ----------
        download_base : str
           File location to store the manifest and the cells.json files

        """
        self.name : str = "allensdk.core.cell_types_cache.CellTypesCache"
        self.download_base : str = init_params.download_base

    @override
    def run(
        self,
        exec_params: CCacheExecParams,
        first_iter: QryItr | None,
        *rest_iter: QryItr | None,
        **params: t.Any,
    ) -> QryItr | None:
        """
        Get the cell types information from allensdk.core.cell_types_cache.CellTypesCache

        Parameters
        ----------
        run_params: dict with the following keys:
            species : list [default: None]
               Should contain one of CellTypesApi.MOUSE or CellTypesApi.HUMAN
            manifest : str [default: manifest.json]
               Name of the manifest json file
            cells : str [default: cells.json]
               Name of the manifest json file

        Returns
        -------
        dict of three elements:
        {
            manifest : manifest file location,
            cells : cells json file location,
            result : list of cell type descriptions; each of which is a dict
        }
        """
        #
        # default_args = {
        #     "species": None,
        #     "mainfest": "manifest.json",
        #     "cells": "cells.json",
        # }
        # rarg = {**default_args, **params} if params else default_args
        #
        _log().debug("CTDbCellCacheQuery Args : %s", str(exec_params))
        self.manifest_file: str = os.path.join(
            self.download_base,
            str(exec_params.manifest)
        )
        self.cells_file : str = os.path.join(
            self.download_base,
            str(exec_params.cells)
        )
        ctc = allensdk.core.cell_types_cache.CellTypesCache(
            manifest_file=self.manifest_file
        )
        ct_list = ctc.get_cells(
            file_name=self.cells_file,
            species=exec_params.species
        )
        _log().debug("CTDbCellCacheQuery CT List : %d", len(ct_list))
        return ct_list

    @override
    @classmethod
    def params_type(cls) -> type[CCacheBaseParams]:
        return cls.QryParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> CCacheBaseParams:
        return cls.QryParams.model_validate(param_dict)


class CellAPIExecParams(CerebrumBaseModel):
    species   : t.Annotated[str | None, Field(title="Species")] = None

CAPIBaseParams : t.TypeAlias = BaseParams[NoneParams, CellAPIExecParams] 

class CTDbCellApiQuery(DbQuery[NoneParams, CellAPIExecParams]):
    class QryParams(CAPIBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[CellAPIExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.name : str = "allensdk.api.queries.cell_types_api.CellTypesApi"

    @override
    def run(
        self,
        exec_params: CellAPIExecParams,
        first_iter: QryItr | None,
        *rest_iter: QryItr | None,
        **params: t.Any
    ) -> QryItr:
        """
        Get the cell types information from allensdk.api.queries.cell_types_api.CellTypesApi

        Parameters
        ----------
        run_params: dict with the following keys:
          species : list [default: None]
             Should contain one of CellTypesApi.MOUSE or CellTypesApi.HUMAN

        Returns
        -------
        dict of on elements:
        {
           result : list
           A list of descriptions of cell types; each description is a dict
        }
        """
        #
        # def_args = {"species": None}
        # rarg = {**def_args, **run_params} if run_params else default_args
        # sp_arg = [rarg["species"]] if rarg["species"] else None
        #
        _log().debug("CTDbCellApiQuery Args : %s", str(exec_params))
        ctxa = allensdk.api.queries.cell_types_api.CellTypesApi()
        ct_list = ctxa.list_cells_api(species=exec_params.species)
        _log().debug("CTDbCellApiQuery CT List : %d", len(ct_list))
        return ct_list

    @override
    @classmethod
    def params_type(cls) -> type[CAPIBaseParams]:
        return cls.QryParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> CAPIBaseParams:
        return cls.QryParams.model_validate(param_dict)



class GlifApiExecParams(CerebrumBaseModel):
    key   : t.Annotated[str, Field(title="Query Key")]
    first : t.Annotated[bool, Field(title="Select Only First")]

GlifApiBaseParams : t.TypeAlias = BaseParams[NoneParams, GlifApiExecParams] 

class CTDbGlifApiQuery(DbQuery[NoneParams, GlifApiExecParams]):
    class QryParams(GlifApiBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[GlifApiExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.name : str = "allensdk.api.queries.glif_api.GlifApi"
        self.glif_api : GlifApi = GlifApi()
        self.key_fn : t.Callable[[t.Any], str] = lambda x: x

    @override
    def run(
        self,
        exec_params: GlifApiExecParams,
        first_iter: QryItr | None,
        *rest_iter: QryItr | None,
        **_params: t.Any
    ) -> QryItr | None:
        """
        Get neuronal models using GlifApi for a given iterator of specimen ids

        Parameters
        ----------
        input_iter : Input Iterator of objects with specimen ids (mandatory)
        params: dict ofthe following keys
        {
            "key"   : Access key to obtain specimen ids (default: None),
            "first" : bool (default: False)
        }

        Returns
        -------
        dict : {spec_id : model}
          glif neuronal_models
        """
        if not first_iter:
            return None
        # default_args = {"first": False, "key": None}
        # rarg = {**default_args, **params} if params else default_args
        _log().debug("CTDbGlifApiQuery Args : %s", str(exec_params))
        if exec_params.key:
            self.key_fn = lambda x: x[exec_params.key]
        if bool(exec_params.first) is False:
            return iter(
                {
                    "ct": x,
                    "glif": self.glif_api.get_neuronal_models(self.key_fn(x)),
                }
                for x in first_iter
                if x
            )
        else:
            return iter(
                {
                    "ct": x,
                    "glif": next(
                        iter(self.glif_api.get_neuronal_models(self.key_fn(x))),
                        None
                    ),
                }
                for x in first_iter
                if x
            )

    @override
    @classmethod
    def params_type(cls) -> type[GlifApiBaseParams]:
        return cls.QryParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> GlifApiBaseParams:
        return cls.QryParams.model_validate(param_dict)


class GAMCExecParams(CerebrumBaseModel):
    suffix    : t.Annotated[str, Field(title="Output Suffix")]
    output_dir: t.Annotated[str, Field(title="Output Dir.")]

GAMCBaseParams : t.TypeAlias = BaseParams[NoneParams, GAMCExecParams] 

class CTDbGlifApiModelConfigQry(DbQuery[NoneParams, GAMCExecParams]):
    class QryParams(GAMCBaseParams):
        init_params: t.Annotated[NoneParams, Field(title='Init Params')]
        exec_params: t.Annotated[GAMCExecParams, Field(title='Exec Params')]

    def __init__(self, init_params: NoneParams, **params: t.Any):
        self.name : str = "allensdk.api.queries.glif_api.GlifApi"
        self.glif_api : GlifApi = GlifApi()

    def download_model_config(self, rcd_dict: dict[str, t.Any]):
        neuronal_model_id = rcd_dict["glif"]["neuronal_models"][0]["id"]
        spec_id = rcd_dict["ct"]["specimen__id"]
        neuron_config = self.glif_api.get_neuron_configs([neuronal_model_id])
        if neuron_config:
            cfg_fname = Path(
                self.output_dir, "{}{}".format(spec_id, self.suffix)
            )
            with open(cfg_fname, "w") as cfg_fx:
                json.dump(neuron_config[neuronal_model_id], cfg_fx, indent=4)
        return rcd_dict

    @override
    def run(
        self,
        exec_params: GAMCExecParams,
        first_iter: QryItr | None,
        *rest_iter: QryItr | None,
        **_params: t.Any
    ) -> QryItr | None:
        if not first_iter:
            return None
        self.suffix : str= exec_params.suffix
        self.output_dir : str = exec_params.output_dir
        _log().debug("CTDbGlifApiQuery Args : %s", str(exec_params))
        # Create Config
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return iter(self.download_model_config(rcx) for rcx in first_iter if rcx)

    @override
    @classmethod
    def params_type(cls) -> type[GAMCBaseParams]:
        return cls.QryParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> GAMCBaseParams:
        return cls.QryParams.model_validate(param_dict)


class DFBuilder:
    TableTypes: t.TypeAlias = t.Literal["ct", "glif", "nm", "nmr"]
    @staticmethod
    def clean_nrn_models(nrn_dct: dict[str, t.Any]):
        return exclude_keys(
            (
                nrn_dct | 
                prefix_keys(
                    nrn_dct['neuronal_model_template'],
                    'template'
                )
            ),
            set([
                'neuronal_model_template',
                'neuronal_model_runs',
                'well_known_files'
            ])
        )
    
    @staticmethod
    def clean_nrn_model_runs(nrn_dict: dict[str, t.Any]):
        return (
            exclude_keys(
                nrx,
                set(['well_known_files'])
            )
            for nrx in nrn_dict["neuronal_model_runs"]
        )
     
    
    @staticmethod
    def nm_row(glf_dct: dict[str, t.Any] | None):
        if glf_dct is None:
            return iter([])
        return (
            DFBuilder.clean_nrn_models(nmx)
            for nmx in glf_dct['neuronal_models']
        )
    
    @staticmethod
    def nmruns_row(glf_dct: dict[str, t.Any] | None):
        if glf_dct is None:
            return iter([])
        return itertools.chain.from_iterable(
            DFBuilder.clean_nrn_model_runs(nmx)    
            for nmx in glf_dct['neuronal_models']
        )
        
    @staticmethod
    def glif_row(glf_dct: dict[str,t.Any]):
        return exclude_keys(
            glf_dct,
            set(["neuronal_models"])
        )
    
    @staticmethod
    def build(
        in_iter: QryItr | None,
        df_source: TableTypes = "ct",
        **_params: t.Any,
    ) -> pl.DataFrame | None:
        if in_iter is None:
            return None
        match df_source:
            case "ct":
                return pl.DataFrame(qrst['ct'] for qrst in in_iter)
            case "glif":
                return pl.DataFrame(
                    (
                        DFBuilder.glif_row(tgx['glif'])
                        for tgx in in_iter if tgx['glif'] is not None
                    )
                )
            case "nm":
                return pl.DataFrame(
                    itertools.chain.from_iterable(
                        DFBuilder.nm_row(tgx['glif'])
                        for tgx in in_iter
                    )
                )
            case "nmr":
                return pl.DataFrame(
                    itertools.chain.from_iterable(
                        DFBuilder.nmruns_row(tgx['glif'])
                        for tgx in in_iter 
                    )
                )


class ABMCTDuckDBWriter(QryDBWriter):
    def __init__(self, db_conn: duckdb.DuckDBPyConnection):
        self.conn : duckdb.DuckDBPyConnection = db_conn

    @override
    def write(
        self,
        in_iter: QryItr | None,
        **_params: t.Any,
    ) -> None:
        for tbx in t.get_args(DFBuilder.TableTypes):
            result_df = DFBuilder.build(in_iter, tbx)  # pyright: ignore[reportUnusedVariable]
            tb_name = f"abm_celltypes_{tbx}"
            self.conn.execute(
                f"CREATE OR REPLACE TABLE {tb_name} AS SELECT * FROM result_df"
            )
        self.conn.commit()
