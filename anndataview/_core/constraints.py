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

