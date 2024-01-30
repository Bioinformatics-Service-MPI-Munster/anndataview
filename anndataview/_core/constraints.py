from __future__ import annotations
from future.utils import string_types
from typing import NoReturn, Union, Optional
from functools import partial
import warnings

import pandas as pd
import numpy as np
from anndata import AnnData

import logging
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


class Constraint(object):
    """\
    A superclass for all AnnData constraints.
    
    Subclass this and override the required methods to provide compatibility with 
    AnnDataView objects.
    
    Parameters
    ----------
    
    See Also
    --------
    AnnDataView
    
    Notes
    -----
    This setup allows for a lot of flexibility in creating reusable contraints
    for AnnDataView objects. Importantly, from_dict and to_dict must produce 
    output consistent with each other to support constraint serialisation in 
    the AnnData object.
    """
    
    __name__ = 'GenericConstraint'
    _call_methods = {}
    plugins = {}
    plugin_methods = {}
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.plugins[f'{cls.__module__}.{cls.__qualname__}'] = cls
        for method_name, axis in cls._call_methods.items():
            cls.plugin_methods[method_name] = cls, axis
    
    def __init__(self) -> None:
        pass
    
    @classmethod
    def new_constraint(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    @classmethod
    def from_dict(cls, constraint_dict) -> Constraint:
        if 'class' not in constraint_dict:
            raise NotImplementedError("Subclasses of Constraint must override 'from_dict'!")
        
        return cls.plugins[constraint_dict['class']].from_dict(constraint_dict)
    
    def _to_dict(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Subclasses of Constraint must override 'to_dict'!")
    
    def to_dict(self, *args, **kwargs) -> dict:
        """\
        Return a dict that describes this constraint.
        """
        base_dict = self._to_dict(*args, **kwargs)
        if 'name' not in base_dict:
            base_dict['name'] = self.__name__
        base_dict['class'] = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        return base_dict
    
    def valid(self, *args, **kwargs) -> np.array:
        """\
        Return a numpy ndarray of bool dtype denoting indices passing this constraint.
        """
        raise NotImplementedError("Subclasses of Constraint must override 'valid'!")
    
    @classmethod
    def _add_to_view(cls, *args, **kwargs) -> Constraint:
        try:
            axis = kwargs.pop("axis", None)
        except KeyError:
            raise KeyError("Must provide axis argument!")
        
        try:
            vdata = kwargs.pop("vdata", None)
        except KeyError:
            raise KeyError("Must provide vdata argument!")
        
        c = cls(*args, **kwargs)
        
        if axis == 0:
            vdata.add_obs_constraint(c)
        elif axis == 1:
            vdata.add_var_constraint(c)
        else:
            raise ValueError("axis must be 0 (obs) or 1 (var)")
        return c


class DataFrameConstraint(Constraint):
    """\
    A superclass for constraints on data frame columns.
    
    Subclass this and override the three methods to provide compatibility with 
    AnnDataView objects.
    
    Parameters
    ----------
    
    See Also
    --------
    AnnDataView
    
    Notes
    -----
    This setup allows for a lot of flexibility in creating reusable contraints
    for AnnDataView objects. Importantly, from_dict and to_dict must produce 
    output consistent with each other to support constraint serialisation in 
    the AnnData object.
    """
    
    __name__ = 'DataFrameConstraint'
    _call_methods = {}
    
    def __init__(self) -> None:
        pass
    
    def filter(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """\
        Filter the provided pandas DataFrame according to this constraint.
        Returns a subset of the 'annotations' DataFrame with rows that pass 
        the 'valid' check.
        """
        return annotations.loc[self.valid(annotations), :]
    
    def test_compatibility(self, adata: AnnData) -> bool:
        return True


class CategoricalDataFrameConstraint(DataFrameConstraint):
    """\
    A constraint on pandas DataFrame categorical dtype columns.
    
    Parameters
    ----------
    key
        A column name denoting a categorical dtype column
    categories
        A list or tuple of strings with category 
    
    See Also
    --------
    DataFrameConstraint
    
    Notes
    -----
    
    """
    
    __name__ = 'CategoricalDataFrameConstraint'
    _call_methods = {
        'add_categorical_obs_constraint': 0,
        'add_categorical_var_constraint': 1,
    }
    
    def __init__(
        self, 
        key: str, 
        categories: Optional[Union[list, tuple, str]] = None,
        invert: bool = False,
    ) -> NoReturn:
        if isinstance(categories, string_types):
            categories = [categories]
        self.key = key
        self.categories = categories
        self.invert = invert
        
    def __repr__(self) -> str:
        categories = ", ".join(self.categories)
        constraint_type = 'excluding' if self.invert else 'with'
        return f"{self.__class__.__qualname__} on key '{self.key}' {constraint_type} categories '{categories}'"
    
    @classmethod
    def from_dict(cls, constraint_dict: dict) -> CategoricalDataFrameConstraint:
        return cls(
            key=constraint_dict.get('key'),
            categories=constraint_dict.get('categories', None),
            invert=constraint_dict.get('invert', False),
        )
    
    def _to_dict(self) -> dict:
        return {
            'key': self.key,
            'categories': self.categories,
            'invert': self.invert,
        }
    
    def valid(self, adata: AnnData, axis:int=0) -> np.array:
        if axis == 0:
            column_data = adata.obs[self.key]
        else:
            column_data = adata.obs[self.key]


        categories = self.categories
        column_categories = column_data.dtype.categories.to_list()
        if categories is None:
            categories = column_categories
        
        # check category consistency
        for category in categories:
            if category not in categories:
                warnings.warn(f"Category '{category}' not found in column categories ({column_categories})!")
        
        if self.invert:
            return ~column_data.isin(categories).to_numpy()
        else:
            return column_data.isin(categories).to_numpy()

    def test_compatibility(self, adata: AnnData) -> bool:
        column_data = adata.obs[self.key]
        
        if column_data.dtype.name != 'category':
            raise ValueError(f"Subsetting currently only works on "
                             f"categorical groups, not {column_data.dtype.name}!")
        
        return True


class NumericalConstraint(Constraint):
    """\
    A constraint on pandas DataFrame numerical dtype columns.
    
    Parameters
    ----------
    key
        A column name denoting a numerical dtype column
    categories
        A list or tuple of strings with category 
    
    See Also
    --------
    DataFrameConstraint
    
    Notes
    -----
    
    """
    
    __name__ = 'NumericalConstraint'
    _call_methods = {
        'add_numerical_obs_constraint': 0,
        'add_numerical_var_constraint': 1,
    }
    
    def __init__(
        self, 
        key: str, 
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        exclusive=True,
    ) -> NoReturn:
        if min_value is None and max_value is None:
            raise ValueError('Please provide min_value, max_value, or both!')
        
        self.key = key
        self.min_value = min_value
        self.max_value = max_value
        self.exclusive = exclusive
        
    def __repr__(self) -> str:
        comparison = '' if self.exclusive else '='
        s = f"{self.__class__.__qualname__} on key '{self.key}'\n"
        if self.min_value is not None and self.max_value is not None:
            s += f'{self.min_value} <{comparison} value <{comparison} {self.max_value}'
        elif self.min_value is not None:
            s += f'{self.min_value} <{comparison} value'
        else:
            s += f'value <{comparison} {self.max_value}'
        return s
            
    @classmethod
    def from_dict(cls, constraint_dict: dict) -> NumericalConstraint:
        return cls(
            key=constraint_dict.get('key'),
            min_value=constraint_dict.get('min_value', None),
            max_value=constraint_dict.get('max_value', None),
            exclusive=constraint_dict.get('exclusive', True),
        )
    
    def _to_dict(self) -> dict:
        return {
            'key': self.key,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'exclusive': self.exclusive,
        }
    
    def valid(self, adata: AnnData, axis:int=0, layer=None) -> np.array:
        if axis == 0:
            column_data = adata.obs_vector(self.key, layer=layer)
        else:
            column_data = adata.var_vector(self.key, layer=layer)
        
        valid = np.array([True] * len(column_data))
        if self.min_value is not None:
            if self.exclusive:
                valid = np.logical_and(valid, column_data > self.min_value)
            else:
                valid = np.logical_and(valid, column_data >= self.min_value)
        
        if self.max_value is not None:
            if self.exclusive:
                valid = np.logical_and(valid, column_data < self.max_value)
            else:
                valid = np.logical_and(valid, column_data <= self.max_value)

        return valid

    def test_compatibility(self, adata: AnnData) -> pd.DataFrame:
        try:
            is_numerical = np.issubdtype(adata.obs_vector(self.key).dtype, np.number)
        except TypeError:
            is_numerical = False
        
        if not is_numerical:
            raise ValueError(f"Column '{self.key}' is not numerical")
        
        return is_numerical


class IndexDataFrameConstraint(DataFrameConstraint):
    """\
    A constraint on pandas DataFrame index.
    
    Parameters
    ----------
    key
        A column name denoting a numerical dtype column
    categories
        A list or tuple of strings with category 
    
    See Also
    --------
    DataFrameConstraint
    
    Notes
    -----
    Constraints can be: index entries, integer indices, 
    or boolean indices
    """
    
    __name__ = 'IndexDataFrameConstraint'
    _call_methods = {
        'add_index_obs_constraint': 0,
        'add_index_var_constraint': 1,
    }
    
    def __init__(
        self, 
        indices: Union[list, tuple, str, int, np.ndarray],
    ) -> NoReturn:        
        # handle case where a single index is given
        if isinstance(indices, (str, int)):
            indices = [indices]

        self.indices = np.array(indices)
        
    def __repr__(self) -> str:
        index_start = ", ".join(self.indices[:5])
        index_ellipsis = ', ...' if len(index_start) > 5 else ''
        s = f"{self.__class__.__qualname__}: {index_start}{index_ellipsis}'"
        return s
            
    @classmethod
    def from_dict(cls, constraint_dict: dict) -> IndexDataFrameConstraint:
        return cls(
            indices=constraint_dict.get('indices'),
        )
    
    def _to_dict(self) -> dict:
        return {
            'indices': [ix for ix in self.indices],
        }
    
    def valid(self, adata: AnnData, axis: int=0) -> np.array:
        if axis == 0:
            return adata.obs.index.isin(self.indices)
        return adata.var.index.isin(self.indices)

    def test_compatibility(self, adata: AnnData, axis: int=0) -> pd.DataFrame:
        if axis == 0:
            annotations = adata.obs
        else:
            annotations = adata.var
            
        try:
            if np.all(annotations.index.isin(self.indices)):
                annotations.loc[self.indices, :]
        except KeyError:
            raise
        
        return True


class ZscoreConstraint(Constraint):
    """\
    A constraint on pandas DataFrame numerical dtype columns
    following Z-score conversion of all values.
    
    Parameters
    ----------
    key
        A column name denoting a numerical dtype column
    min_value
        The minimum Z-score
    max_value
        The maximum Z-score
    exclusive
        If True, the comparison to min and max values is > / <,
        if False it is >= / <=
    include_zeros
        If set to False, exclude all values that are 0 in Z-score 
        transform
    
    See Also
    --------
    
    
    Notes
    -----
    
    """
    
    __name__ = 'ZscoreConstraint'
    _call_methods = {
        'add_zscore_obs_constraint': 0,
        'add_zscore_var_constraint': 1,
    }
    
    def __init__(
        self, 
        key: str, 
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        include_zeros: bool = True,
        exclusive: bool = True,
    ) -> NoReturn:
        if min_value is None and max_value is None:
            raise ValueError('Please provide min_value, max_value, or both!')
        
        self.key = key
        self.min_value = min_value
        self.max_value = max_value
        self.exclusive = exclusive
        self.include_zeros = include_zeros
        
    def __repr__(self) -> str:
        comparison = '' if self.exclusive else '='
        s = f"{self.__class__.__qualname__} on key '{self.key}'\n"
        if self.min_value is not None and self.max_value is not None:
            s += f'{self.min_value} <{comparison} value <{comparison} {self.max_value}'
        elif self.min_value is not None:
            s += f'{self.min_value} <{comparison} value'
        else:
            s += f'value <{comparison} {self.max_value}'
        if not self.include_zeros:
            s += "\nExcluding 0s in Z-score transform."
        return s
            
    @classmethod
    def from_dict(cls, constraint_dict: dict) -> ZscoreConstraint:
        return cls(
            key=constraint_dict.get('key'),
            min_value=constraint_dict.get('min_value', None),
            max_value=constraint_dict.get('max_value', None),
            include_zeros=constraint_dict.get('include_zeros', None),
            exclusive=constraint_dict.get('exclusive', True),
        )
    
    def _to_dict(self) -> dict:
        return {
            'key': self.key,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'include_zeros': self.include_zeros,
            'exclusive': self.exclusive,
        }
    
    def valid(self, adata: AnnData, axis:int=0, layer=None) -> np.array:
        if axis == 0:
            column_data = adata.obs_vector(self.key, layer=layer)
        else:
            column_data = adata.var_vector(self.key, layer=layer)
        
        if not self.include_zeros:
            non_zero = column_data[column_data > 0]
        else:
            non_zero = np.repeat(True, len(column_data))

        mean = np.nanmean(column_data[non_zero])
        sd = np.nanstd(column_data[non_zero])

        zscores = (column_data - mean) / sd
        zscores[~non_zero] = np.nan
        
        valid = np.array([True] * len(zscores))
        if self.min_value is not None:
            if self.exclusive:
                valid = np.logical_and(valid, zscores > self.min_value)
            else:
                valid = np.logical_and(valid, zscores >= self.min_value)
        
        if self.max_value is not None:
            if self.exclusive:
                valid = np.logical_and(valid, zscores < self.max_value)
            else:
                valid = np.logical_and(valid, zscores <= self.max_value)

        return valid

    def test_compatibility(self, adata: AnnData, layer=None) -> pd.DataFrame:
        try:
            is_numerical = np.issubdtype(adata.obs_vector(self.key, layer=layer).dtype, np.number)
        except TypeError:
            is_numerical = False
        
        if not is_numerical:
            raise ValueError(f"Column '{self.key}' is not numerical")
        
        return is_numerical
