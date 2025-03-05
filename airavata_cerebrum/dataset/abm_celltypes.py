import json
import logging
import os
import typing as t
import traitlets
import allensdk.core.cell_types_cache
import allensdk.api.queries.cell_types_api
#
from typing_extensions import override
from allensdk.api.queries.glif_api import GlifApi
from pathlib import Path
from ..base import DbQuery, QryItr, OpXFormer


def _log():
    return logging.getLogger(__name__)


class CTDbCellCacheQuery(DbQuery):
    @t.final
    class QryTraits(traitlets.HasTraits):
        download_base = traitlets.Unicode()
        species = traitlets.Unicode()
        manifest = traitlets.Unicode()
        cells = traitlets.Unicode()

    def __init__(self, **params: t.Any):
        """
        Initialixe Cache Query
        Parameters
        ----------
        download_base : str
           File location to store the manifest and the cells.json files

        """
        self.name : str = "allensdk.core.cell_types_cache.CellTypesCache"
        self.download_base : str = params["download_base"]

    @override
    def run(
        self,
        in_iter: QryItr | None,
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
        default_args = {
            "species": None,
            "mainfest": "manifest.json",
            "cells": "cells.json",
        }
        rarg = {**default_args, **params} if params else default_args
        #
        _log().debug("CTDbCellCacheQuery Args : %s", rarg)
        self.manifest_file: str = os.path.join(
            self.download_base,
            str(rarg["mainfest"])
        )
        self.cells_file : str = os.path.join(
            self.download_base,
            str(rarg["cells"])
        )
        ctc = allensdk.core.cell_types_cache.CellTypesCache(
            manifest_file=self.manifest_file
        )
        ct_list = ctc.get_cells(file_name=self.cells_file, species=rarg["species"])
        _log().debug("CTDbCellCacheQuery CT List : %d", len(ct_list))
        return ct_list

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.QryTraits


class CTDbCellApiQuery(DbQuery):
    @t.final
    class QryTraits(traitlets.HasTraits):
        species = traitlets.Unicode()

    def __init__(self, **params: t.Any):
        self.name : str = "allensdk.api.queries.cell_types_api.CellTypesApi"

    @override
    def run(
        self,
        in_iter: QryItr | None,
        **run_params: t.Any
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
        default_args = {"species": None}
        rarg = {**default_args, **run_params} if run_params else default_args
        sp_arg = [rarg["species"]] if rarg["species"] else None
        #
        _log().debug("CTDbCellApiQuery Args : %s", rarg)
        ctxa = allensdk.api.queries.cell_types_api.CellTypesApi()
        ct_list = ctxa.list_cells_api(species=sp_arg)
        _log().debug("CTDbCellApiQuery CT List : %d", len(ct_list))
        return ct_list

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.QryTraits


class CTDbGlifApiQuery(DbQuery):
    @t.final
    class QryTraits(traitlets.HasTraits):
        key = traitlets.Unicode()
        first = traitlets.Bool()

    def __init__(self, **params: t.Any):
        self.name : str = "allensdk.api.queries.glif_api.GlifApi"
        self.glif_api : GlifApi = GlifApi()
        self.key_fn : t.Callable[[t.Any], str] = lambda x: x

    @override
    def run(
        self,
        in_iter: QryItr | None,
        **params: t.Any
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
        if not in_iter:
            return None
        default_args = {"first": False, "key": None}
        rarg = {**default_args, **params} if params else default_args
        _log().debug("CTDbGlifApiQuery Args : %s", rarg)
        if rarg["key"]:
            self.key_fn = lambda x: x[rarg["key"]]
        if bool(rarg["first"]) is False:
            return iter(
                {
                    "ct": x,
                    "glif": self.glif_api.get_neuronal_models(self.key_fn(x)),
                }
                for x in in_iter
                if x
            )
        else:
            return iter(
                {
                    "ct": x,
                    "glif": next(
                        iter(self.glif_api.get_neuronal_models(self.key_fn(x))), None
                    ),
                }
                for x in in_iter
                if x
            )

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.QryTraits


class CTDbGlifApiModelConfigQry(DbQuery):
    @t.final
    class QryTraits(traitlets.HasTraits):
        suffix = traitlets.Unicode()
        output_dir = traitlets.Unicode()

    def __init__(self, **params: t.Any):
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
        in_iter: QryItr | None,
        **params: t.Any
    ) -> QryItr | None:
        if not in_iter:
            return None
        self.suffix : str= params["suffix"]
        self.output_dir : str = params["output_dir"]
        _log().debug("CTDbGlifApiQuery Args : %s", params)
        # Create Config
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return iter(self.download_model_config(rcx) for rcx in in_iter if rcx)

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.QryTraits


#
# ------- Query and Xform Registers -----
#
def query_register() -> list[type[DbQuery]]:
    return [
        CTDbCellCacheQuery,
        CTDbCellApiQuery,
        CTDbGlifApiQuery,
        CTDbGlifApiModelConfigQry,
    ]


def xform_register() -> list[type[OpXFormer]]:
    return []
