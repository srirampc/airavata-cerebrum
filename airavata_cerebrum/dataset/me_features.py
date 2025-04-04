import typing as t
import duckdb
import polars as pl

from typing_extensions import override
from pydantic import Field

from ..base import (
    CerebrumBaseModel, BaseParams,
    DbQuery, QryDBWriter, QryItr
)

class InitParams(CerebrumBaseModel):
    file_url : t.Annotated[str, Field(title="Download Base")]

class ExecParams(CerebrumBaseModel):
    key           : t.Annotated[str , Field(title="Input Key")] = ""
    attribute     : t.Annotated[str | None, Field(title="Attribute")] = None
    mef_attribute  : t.Annotated[str | None , Field(title="MEF Property")] = None


MEFParamsBase : t.TypeAlias = BaseParams[InitParams,  ExecParams] 

class MEFDataQuery(DbQuery[InitParams, ExecParams]):
    class QryParams(MEFParamsBase):
        init_params: t.Annotated[InitParams, Field(title='Init Params')]
        exec_params: t.Annotated[ExecParams, Field(title='Exec Params')]
    
    def __init__(self, init_params: InitParams, **params: t.Any):
        self.file_url : str = init_params.file_url

    def find_in_def(
        self,
        medf: pl.DataFrame,
        column: str | None,
        cvalue: t.Any
    ) -> dict[str, t.Any]:
        if column is None:
            return {}
        fdf = medf.filter(medf.get_column(column) == cvalue).head(1)
        if fdf.shape[0] == 0:
            return {}
        return fdf.to_dicts()[0]

    @override
    def run(
        self,
        exec_params: ExecParams,
        first_iter: QryItr | None,
        *rest_iter: QryItr | None,
        **params: t.Any,
    ) -> QryItr | None:
        edf = pl.read_excel(self.file_url)
        if first_iter is None:
            return ( {"mef": rcdx} for rcdx in edf.to_dicts() )
        #
        def select_fn(initx: t.Any):
            return (
                initx[exec_params.key][exec_params.attribute]
                if exec_params.key else initx[exec_params.attribute]
            )
        #
        return (
            inx | {
                "mef" : self.find_in_def(
                    edf,
                    exec_params.mef_attribute,
                    select_fn(inx)
                )
            }
            for inx in first_iter
        )
        

    @override
    @classmethod
    def params_type(cls) -> type[MEFParamsBase]:
        return cls.QryParams

    @override
    @classmethod
    def params_instance(cls, param_dict: dict[str, t.Any]) -> MEFParamsBase:
        return cls.QryParams.model_validate(param_dict)


class DFBuilder:
    @staticmethod
    def build(
        in_iter: QryItr | None,
        **_params: t.Any,
    ) -> pl.DataFrame | None:
        if in_iter is None:
            return None
        return pl.DataFrame(qrst['mef'] for qrst in in_iter)


class MEFDuckDBWriter(QryDBWriter):
    def __init__(self, db_conn: duckdb.DuckDBPyConnection):
        self.conn : duckdb.DuckDBPyConnection = db_conn

    @override
    def write(
        self,
        in_iter: QryItr | None,
        **_params: t.Any,
    ) -> None:
        result_df = DFBuilder.build(in_iter)  # pyright: ignore[reportUnusedVariable]
        tb_name = f"mef_features"
        self.conn.execute(
            f"CREATE OR REPLACE TABLE {tb_name} AS SELECT * FROM result_df"
        )
        self.conn.commit()
