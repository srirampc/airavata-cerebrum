import typing as t
#

def class_qual_name(src_class: type):
    return  ".".join([src_class.__module__, src_class.__name__])


def exclude_keys(
    r_dct: dict[str, t.Any],
    key_set: set[str]
) -> dict[str, t.Any]:
    return {kx: vx for kx, vx in r_dct.items() if kx not in key_set}


def prefix_keys(
    r_dct: dict[str, t.Any],
    pfx: str,
    sep:str = '_'
) -> dict[str, t.Any]:
    return {f"{pfx}{sep}{kx}": vx for kx, vx in r_dct.items()}

