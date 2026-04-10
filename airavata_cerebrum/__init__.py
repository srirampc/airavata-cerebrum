import numpy.typing as npt
import numpy as np
import typing as t

NPFloat : t.TypeAlias = np.floating[t.Any]
NPInteger : t.TypeAlias = np.integer[t.Any]
NDFloatArray : t.TypeAlias = npt.NDArray[np.floating[t.Any]]
NDIntArray : t.TypeAlias = npt.NDArray[np.integer[t.Any]]
NDBoolArray : t.TypeAlias = npt.NDArray[np.bool_]
NDObjectArray : t.TypeAlias = npt.NDArray[np.object_]
NDAnyArray: t.TypeAlias = npt.NDArray[t.Any]
FloatT: t.TypeAlias = float | NPFloat
IntegerT: t.TypeAlias = int | NPInteger
NPDType: t.TypeAlias = npt.DTypeLike
