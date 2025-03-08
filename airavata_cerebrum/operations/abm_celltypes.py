import traitlets
import typing as t
from typing_extensions import override
#
from ..base import OpXFormer, XformItr, DbQuery
from .json_filter import IterJPatchFilter, IterJPointerFilter
from .dict_filter import IterAttrFilter


class CTModelNameFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        model_name = traitlets.Unicode()

    def __init__(self, **params: t.Any):
        self.name : str = __name__ + ".CTModelNameFilter"
        self.filter_fmt : str = "$.glif.neuronal_models[?('{}' in @.name)]"
        self.dest_path : str = "/glif/neuronal_models"
        self.jpatch_filter : IterJPatchFilter = IterJPatchFilter(**params)

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        model_name = params["model_name"]
        filter_exp = self.filter_fmt.format(model_name)
        return self.jpatch_filter.xform(in_iter,
                                        filter_exp=filter_exp,
                                        dest_path=self.dest_path)

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.FilterTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.FilterTraits(**trait_values)


class CTExplainedRatioFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        ratio = traitlets.Float()

    def __init__(self, **params: t.Any):
        self.name : str = __name__ + ".CTExplainedRatioFilter"
        self.filter_fmt : str = "$.glif.neuronal_models[0].neuronal_model_runs[?(@.explained_variance_ratio > {})]"
        self.dest_path : str = "/glif/neuronal_models/0/neuronal_model_runs"
        self.final_path : str = "/glif/neuronal_models/0/neuronal_model_runs/0"
        self.jpatch_filter : IterJPatchFilter = IterJPatchFilter(**params)
        self.jptr_filter : IterJPointerFilter = IterJPointerFilter(**params)

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any,
    ) -> XformItr | None:
        ratio_value = params["ratio"]
        filter_exp = self.filter_fmt.format(ratio_value)
        patch_out = self.jpatch_filter.xform(in_iter,
                                             filter_exp=filter_exp,
                                             dest_path=self.dest_path)
        return self.jptr_filter.xform(patch_out,
                                      path=self.final_path)

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.FilterTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.FilterTraits(**trait_values)


class CTPropertyFilter(OpXFormer):
    @t.final
    class FilterTraits(traitlets.HasTraits):
        key = traitlets.Unicode()
        region = traitlets.Unicode()
        layer = traitlets.Unicode()
        line = traitlets.Unicode()
        reporter_status = traitlets.Unicode()

    QUERY_FILTER_MAP : dict[str, list[str]] = {
        "region": ["structure_parent__acronym", "__eq__"],
        "layer": ["structure__layer", "__eq__"],
        "line": ["line_name", "__contains__"],
        "reporter_status": ["cell_reporter_status", "__eq__"]
    }

    def __init__(self, **params: t.Any):
        self.cell_attr_filter : IterAttrFilter = IterAttrFilter(**params)

    @override
    def xform(
        self,
        in_iter: XformItr | None,
        **params: t.Any
    ) -> XformItr | None:
        key = params["key"] if "key" in params else None
        filters = []
        for pkey, valx in params.items():
            if pkey in CTPropertyFilter.QUERY_FILTER_MAP:
                filter_attr = CTPropertyFilter.QUERY_FILTER_MAP[pkey].copy()
                filter_attr.append(str(valx))
                filters.append(filter_attr)
        return self.cell_attr_filter.xform(in_iter,
                                           key=key,
                                           filters=filters)

    @override
    @classmethod
    def trait_type(cls) -> type[traitlets.HasTraits]:
        return cls.FilterTraits

    @override
    @classmethod
    def trait_instance(cls, **trait_values: t.Any) -> traitlets.HasTraits:
        return cls.FilterTraits(**trait_values)

#
# ------- Query and Xform Registers -----
#
def query_register() -> list[type[DbQuery]]:
    return []


def xform_register() -> list[type[OpXFormer]]:
    return [
        CTModelNameFilter,
        CTExplainedRatioFilter,
        CTPropertyFilter,
    ]
