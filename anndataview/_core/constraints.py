from __future__ import annotations
from future.utils import string_types
from typing import Union, Optional
import warnings

import pandas as pd
import numpy as np

import logging
logging.captureWarnings(True)
logger = logging.getLogger(__name__)



class DataFrameConstraint(object):
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
    
    def __init__(self) -> None:
        pass
    
    @classmethod
    def from_dict(cls, constraint_dict) -> DataFrameConstraint:
        raise NotImplementedError("Subclasses of DataFrameConstraint must override 'from_dict'!")
    
    def _to_dict(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Subclasses of DataFrameConstraint must override 'to_dict'!")
    
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
        Return a numpy ndarray of bool dtype denoting if a DataFrame Series item 
        passes this constraint.
        """
        raise NotImplementedError("Subclasses of DataFrameConstraint must override 'valid'!")
    
    def filter(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """\
        Filter the provided pandas DataFrame according to this constraint.
        Returns a subset of the 'annotations' DataFrame with rows that pass 
        the 'valid' check.
        """
        return annotations.loc[self.valid(annotations), :]


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
    
    def __init__(
        self, 
        key: str, 
        categories: Optional[Union[list, tuple, str]] = None
    ) -> None:
        if isinstance(categories, string_types):
            categories = [categories]
        self.key = key
        self.categories = categories
        
    def __repr__(self) -> str:
        categories = ", ".join(self.categories)
        return f"{self.__class__.__qualname__} on key '{self.key}' with categories '{categories}'"
    
    @classmethod
    def from_dict(cls, constraint_dict: dict) -> CategoricalDataFrameConstraint:
        return cls(
            key=constraint_dict.get('key'),
            categories=constraint_dict.get('categories', None),
        )
    
    def _to_dict(self) -> dict:
        return {
            'key': self.key,
            'categories': self.categories,
        }
    
    def valid(self, annotations: pd.DataFrame) -> np.array:
        column_data = annotations[self.key]
        
        if column_data.dtype.name != 'category':
            raise ValueError(f"Subsetting currently only works on "
                             f"categorical groups, not {column_data.dtype.name}!")

        categories = self.categories
        column_categories = column_data.dtype.categories.to_list()
        if categories is None:
            categories = column_categories
        
        # check category consistency
        for category in categories:
            if category not in categories:
                warnings.warn(f"Category '{category}' not found in column categories ({column_categories})!")
        
        return column_data.isin(categories).to_numpy()


