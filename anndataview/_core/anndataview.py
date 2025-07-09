from __future__ import annotations
from typing import NoReturn, Optional, Union, Tuple
from copy import deepcopy

from anndata import AnnData
from anndata._core.aligned_mapping import Layers
import numpy as np
import scipy.sparse
import pandas as pd
import json
from deepdiff import DeepDiff

from .annotations import DataFrameView, callable_matcher, ViewDict
from .constraints import Constraint, IndexDataFrameConstraint
from .utils import array_equal_sparse, my_partialmethod, difference_dict

Index1D = Union[slice, int, str, np.int64, np.ndarray]
Index = Union[Index1D, Tuple[Index1D, Index1D], scipy.sparse.spmatrix]


def _add_constraint(self, *args, **kwargs):
    try:
        axis = kwargs.pop("axis", None)
    except KeyError:
        raise KeyError("Must provide axis argument!")
    
    try:
        constraint_class = kwargs.pop("constraint_class", None)
    except KeyError:
        raise KeyError("Must provide constraint_class argument!")
    
    c = constraint_class(*args, **kwargs)
    
    if axis == 0:
        ret = self.add_obs_constraint(c)
    elif axis == 1:
        ret = self.add_var_constraint(c)
    else:
        raise ValueError("axis must be 0 (obs) or 1 (var)")
    return ret


class AnnDataView(object):
    __slots__ = ["_parent_adata", "obs_constraints", "var_constraints"]
        
    def __init__(self, adata, obs_constraints=None, var_constraints=None):
        object.__setattr__(self, "_parent_adata", adata)
        object.__setattr__(self, "obs_constraints", [] if obs_constraints is None else obs_constraints)
        object.__setattr__(self, "var_constraints", [] if var_constraints is None else var_constraints)
    
    #
    # proxying (special cases)
    #
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_parent_adata"), name)
    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_parent_adata"), name)
    def __setattr__(self, name, value):
        if name in ['obs_constraints', 'var_constraints']:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_parent_adata"), name, value)
    
    def __nonzero__(self):
        return bool(object.__getattribute__(self, "_parent_adata"))
    def __str__(self):
        return str(object.__getattribute__(self, "_parent_adata"))
    def __repr__(self):
        return repr(object.__getattribute__(self, "_parent_adata"))
    
    #
    # factories
    #
    _special_names = [
        '__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__', 
        '__contains__', '__delitem__', '__delslice__', '__div__', '__divmod__', 
        '__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__', 
        '__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__',
        '__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__', 
        '__imul__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', 
        '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', 
        '__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', 
        '__neg__', '__oct__', '__or__', '__pos__', '__pow__', '__radd__', 
        '__rand__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__', 
        '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__', 
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', 
        '__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__', 
        '__truediv__', '__xor__', 'next',
    ]
    
    @classmethod
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class"""
        
        def make_method(name):
            def method(self, *args, **kw):
                return getattr(object.__getattribute__(self, "_parent_adata"), name)(*args, **kw)
            return method
        
        namespace = {}
        for name in cls._special_names:
            if hasattr(theclass, name):
                namespace[name] = make_method(name)
        return type("%s(%s)" % (cls.__name__, theclass.__name__), (cls,), namespace)

    
    def __new__(cls, obj, *args, **kwargs):
        """
        creates an proxy instance referencing `obj`. (obj, *args, **kwargs) are
        passed to this class' __init__, so deriving classes can define an 
        __init__ method of their own.
        note: _class_proxy_cache is unique per deriving class (each deriving
        class must hold its own cache)
        """
        try:
            cache = cls.__dict__["_class_proxy_cache"]
        except KeyError:
            cls._class_proxy_cache = cache = {}
        try:
            theclass = cache[obj.__class__]
        except KeyError:
            cache[obj.__class__] = theclass = cls._create_class_proxy(obj.__class__)
        
        ins = object.__new__(theclass)
        theclass.__init__(ins, obj, *args, **kwargs)
        return ins
    

    def __repr__(self) -> str:
        original_lines = self._parent_adata.__repr__().split("\n")
        original_lines[0] = f'{self.__class__.__qualname__} object with n_obs x n_var = {self.n_obs} x {self.n_vars}'
        s = "\n".join(original_lines) + "\n"
        s += '    obs constraints ({}):\n'.format(len(self.obs_constraints))
        for c in self.obs_constraints:
            s += '        ' +  repr(c) + "\n"
        s += '    var constraints ({}):\n'.format(len(self.var_constraints))
        for c in self.var_constraints:
            s += '        ' + repr(c) + "\n"
        s += f"Original n_obs x n_var = {self._parent_adata.n_obs} x {self._parent_adata.n_vars}\n"
        return s
        
    def add_obs_constraint(self, constraint: Union(dict, Constraint)) -> AnnDataView:
        """\
        Add a constraint on the observations of this AnnData object.
        """
        if isinstance(constraint, dict):
            constraint = Constraint.from_dict(constraint)
        constraint.test_compatibility(self._parent_adata)
        self.obs_constraints.append(constraint)
        return self
    
    @property
    def has_obs_constraints(self) -> bool:
        """\
        Check if this view has any constraints on observations.
        
        Returns
        -------
        A bool - True if there are observation constraints, False if not
        """
        return len(self.obs_constraints) > 0
        
    def add_var_constraint(self, constraint: Union(dict, Constraint)) -> AnnDataView:
        """\
        Add a constraint on the variables of this AnnData object.
        """
        if isinstance(constraint, dict):
            constraint = Constraint.from_dict(constraint)
        constraint.test_compatibility(self._parent_adata)
        self.var_constraints.append(constraint)
        return self
    
    @property
    def has_var_constraints(self) -> bool:
        """\
        Check if this view has any constraints on variables.
        
        Returns
        -------
        A bool - True if there are variable constraints, False if not
        """
        return len(self.var_constraints) > 0
    
    def add_view_constraint(self, view_key):
        """
        Add all constraints from an existing view.
        """
        view_info = self.view_info(view_key)
        for obs_constraint in view_info['obs_constraints']:
            print(obs_constraint)
            self.add_obs_constraint(obs_constraint)
        for var_constraint in view_info['var_constraints']:
            print(var_constraint)
            self.add_var_constraint(var_constraint)
        return self
    
    @property
    def has_constraints(self) -> bool:
        """\
        Check if this view has any constraints.
        
        Returns
        -------
        A bool - True if there are constraints, False if not
        """
        return self.has_obs_constraints or self.has_var_constraints
    
    def clear_constraints(self) -> NoReturn:
        """\
        Remove all constraints from this view (variable and observations).
        
        Notes
        -----
        Also resets the view_name, if it was set.
        """
        self.obs_constraints = []
        self.var_constraints = []
    
    def _valid(self, constraints: list, axis: int) -> np.array:
        valid = np.array([True] * self._parent_adata.shape[axis])

        for constraint in constraints:
            v = constraint.valid(self._parent_adata, axis=axis)
            valid = np.logical_and(v, valid)
        return valid
        
    @property
    def valid_obs(self) -> np.array:
        """\
        Determine which observations are valid given the views constraints.
        
        Returns
        -------
        :class:np.ndarray of bool dtype, where valid observations correspond 
        to True, invalid observations to False.
        """
        return self._valid(self.obs_constraints, 0)

    @property
    def valid_obs_ixs(self):
        """\
        Determine indices of valid observations given view constraints.
        """
        return np.ravel(np.argwhere(self.valid_obs))
    
    @property
    def valid_var(self) -> np.array:
        """\
        Determine which variables are valid given the views constraints.
        
        Returns
        -------
        :class:np.ndarray of bool dtype, where valid variables correspond 
        to True, invalid variables to False.
        """
        return self._valid(self.var_constraints, 1)

    @property
    def valid_var_ixs(self):
        """\
        Determine indices of valid variables given view constraints.
        """
        return np.ravel(np.argwhere(self.valid_var))
    
    @property
    def _valid_indices(self):
        oidx = slice(None, None, None)
        vidx = slice(None, None, None)
        
        if self.has_obs_constraints:
            oidx = self.valid_obs_ixs
        if self.has_var_constraints:
            vidx = self.valid_var_ixs
        return (oidx, vidx)
    
    def _adata_subset(self, copy=False) -> AnnData:
        oidx, vidx = self._valid_indices
        adata_view = AnnDataSubset(self._parent_adata, 
                                   oidx=oidx, vidx=vidx, asview=True,
                                   parent_adata=self._parent_adata,
                                   var_constraints=self.var_constraints,
                                   obs_constraints=self.obs_constraints,)
        
        if copy:
            return adata_view.copy()
        return adata_view
    
    @property
    def adata_view(self) -> AnnDataSubset:
        """\
        Obtain the current view as a modifiable AnnData object 
        with information on the parent AnnData and constraints.
        
        The workflow for creating, modifying, and saving views is:
        
        - Create an AnnDataView object from an existing AnnData
        - [Optional] Add constraints to the AnnDataView object
        - Get a modifiable copy of the view using this property
        - Modify the copy as you want
        - Save back the copy modification to the original 
          AnnDataView
        """
        return self._adata_subset(copy=True)
    
    def restore_view(self, view_key):
        view_info = self.view_info(view_key)
        vdata_copy = AnnDataView(self._parent_adata)
        vdata_copy.obs_constraints = view_info['obs_constraints']
        vdata_copy.var_constraints = view_info['var_constraints']
        view_adata = AnnDataSubset(
            X=vdata_copy.X,
            layers=vdata_copy.layers(view_key=view_key),
            obs = vdata_copy.obs(view_key=view_key),
            var = vdata_copy.var(view_key=view_key),
            uns = vdata_copy.uns(view_key=view_key),
            obsm = vdata_copy.obsm(view_key=view_key),
            varm = vdata_copy.varm(view_key=view_key),
            obsp = vdata_copy.obsp(view_key=view_key),
            varp = vdata_copy.varp(view_key=view_key),
            obs_constraints = view_info['obs_constraints'],
            var_constraints = view_info['var_constraints'],
        )
        return view_adata
    
    @property
    def obs(self):
        if self.has_obs_constraints:
            obs_df =  self._parent_adata.obs[self.valid_obs]
        else:
            obs_df =  self._parent_adata.obs
        obs_df.__class__ = DataFrameView
        return obs_df

    @property
    def var(self):
        if self.has_var_constraints:
            var_df = self._parent_adata.var[self.valid_var]
        else:
            var_df = self._parent_adata.var
        var_df.__class__ = DataFrameView
        return var_df

    @property
    def n_obs(self) -> int:
        return self.obs.shape[0]
    
    @property
    def n_vars(self) -> int:
        return self.var.shape[0]
    
    @property
    def shape(self) -> tuple:
        return self.n_obs, self.n_vars

    @property
    def X(self):
        if self._parent_adata.X is not None and self.has_constraints:
            obs_indices, var_indices = self._valid_indices
            return self._parent_adata.X[obs_indices][:, var_indices]
        else:
            return self._parent_adata.X
    
    def _restore_subset_dict_callable(self, original):
        if original.__class__ not in callable_matcher.values():
            original.__class__ = callable_matcher[original.__class__]
        return original
    
    @property
    def layers(self):
        if self.has_constraints:
            layers = self._adata_subset().layers
        else:
            layers = self._parent_adata.layers
        return self._restore_subset_dict_callable(layers)
    
    @property
    def obsm(self):
        if self.has_obs_constraints:
            obsm = self._adata_subset().obsm
        else:
            obsm = self._parent_adata.obsm
        return self._restore_subset_dict_callable(obsm)
    
    @property
    def obsp(self):
        if self.has_obs_constraints:
            obsp =  self._adata_subset().obsp
        else:
            obsp = self._parent_adata.obsp
        return self._restore_subset_dict_callable(obsp)
    
    @property
    def varm(self):
        if self.has_var_constraints:
            varm = self._adata_subset().varm
        else:
            varm = self._parent_adata.varm
        return self._restore_subset_dict_callable(varm)
    
    @property
    def varp(self):
        if self.has_var_constraints:
            varp = self._adata_subset().varp
        else:
            varp = self._parent_adata.varp
        return self._restore_subset_dict_callable(varp)
    
    @property
    def uns(self):
        uns_dict = self._parent_adata.uns
        
        # print('CLASS: ', uns_dict.__class__)
        # uns_dict.__class__ = OverloadedDictView
        return ViewDict(uns_dict)
    
    def obs_vector(self, k: str, *, layer: Optional[str] = None, 
                   view_key: Optional[str] = None) -> np.ndarray:
        if view_key is not None and f'__view__{view_key}__{k}' in self._parent_adata.obs_keys():
            ov = self._parent_adata.obs_vector(f'__view__{view_key}__{k}', layer=layer)
        else:
            ov = self._parent_adata.obs_vector(k, layer=layer)
        
        if not self.has_obs_constraints:
            return ov
        return ov[self.valid_obs]
    
    def var_vector(self, k: str, *, layer: Optional[str] = None, 
                   view_key: Optional[str] = None) -> np.ndarray:
        if view_key is not None and f'__view__{view_key}__{k}' in self._parent_adata.var_keys():
            vv = self._parent_adata.var_vector(f'__view__{view_key}__{k}', layer=layer)
        else:
            vv = self._parent_adata.var_vector(k, layer=layer)
            
        if not self.has_var_constraints:
            return vv
        return vv[self.valid_var]
    
    def _assert_view_index_compatible(self, view, axis):
        if axis == 0:
            annotation_key = 'obs'
        elif axis == 1:
            annotation_key = 'var'
        else:
            raise ValueError("axis can only b 0 (obs) or 1 (var)")
        
        index_incompatible_ixs = ~getattr(view, annotation_key).index.isin(getattr(self._parent_adata, annotation_key).index)
        index_incompatible = getattr(view, annotation_key).index[index_incompatible_ixs]
        if len(index_incompatible) > 0:
            raise ValueError("Some or all indices of view not present "
                             "in original index: {}{}".format(", ".join(index_incompatible[:5]),
                                                              ', ...' if len(index_incompatible) > 5 else ''))
    
    def _add_view_annotations(self, parent_df, view_df, key):
        # compare df columns
        parent_df_subset = parent_df.loc[view_df.index, :]
        diff_columns = []
        for column in view_df.columns:
            if column.startswith('__'):
                continue
            if (
                column not in parent_df_subset or 
                not parent_df_subset[column].astype('object')
                    .equals(view_df[column].astype('object'))
            ):
                diff_columns.append(column)
                
        for diff_column in diff_columns:
            column_name = f'__view__{key}__{diff_column}'
            if pd.api.types.is_numeric_dtype(view_df[diff_column]):
                parent_df[column_name] = np.nan
                parent_df.loc[view_df.index, column_name] = view_df[diff_column]
                parent_df[column_name] = parent_df[column_name].astype(
                    view_df[diff_column].dtype
                )
            else:
                parent_df[column_name] = 'NA'
                parent_df.loc[view_df.index, column_name] = view_df[diff_column]

                if pd.api.types.is_categorical_dtype(view_df[diff_column]):
                    categories = view_df[diff_column].cat.categories.to_list()
                    if 'NA' not in categories and parent_df.shape[0] != view_df.shape[0]:
                        categories.append('NA')

                    parent_df[column_name] = pd.Categorical(
                        parent_df[column_name],
                        categories=categories,
                    )
    
    def add_view_obs(self, vdata, key):
        self._assert_view_index_compatible(vdata, axis=0)
        self._add_view_annotations(self._parent_adata.obs, vdata.obs, key)
    
    def add_view_var(self, vdata, key):
        self._assert_view_index_compatible(vdata, axis=1)
        self._add_view_annotations(self._parent_adata.var, vdata.var, key)
        
    def _ix_converter(self, vdata, axis):
        annotation_key = 'obs' if axis == 0 else 'var'
        
        parent_annotations = getattr(self._parent_adata, annotation_key)
        view_annotations = getattr(vdata, annotation_key)

        parent_ixs = np.where(parent_annotations.index.isin(view_annotations.index))[0]
        parent_index_order = parent_annotations.index[parent_ixs]
        ix_df = pd.DataFrame({'ix': np.array(range(vdata.shape[axis]))}, index=view_annotations.index)
        view_ixs = ix_df.loc[parent_index_order, 'ix'].to_numpy()
        
        return parent_ixs, view_ixs
    
    def _add_view_m(self, vdata, axis, key, matrix_type='m', **kwargs):
        self._assert_view_index_compatible(vdata, axis=axis)
        annotation_key = 'obs' if axis == 0 else 'var'
        
        n_rows = self._parent_adata.shape[axis]
        if matrix_type == 'm':
            parent_ixs, view_ixs = self._ix_converter(vdata, axis=axis)
        elif matrix_type == 'p':
            parent_ixs_axis, view_ixs_axis = self._ix_converter(vdata, axis=axis)
            parent_ixs = np.ix_(parent_ixs_axis, parent_ixs_axis)
            view_ixs = np.ix_(view_ixs_axis, view_ixs_axis)
        else:
            raise ValueError(f"Wrong matrix type {matrix_type} (m or p possible)")
        
        vdata_mm = getattr(vdata, f'{annotation_key}{matrix_type}')
        parent_mm = getattr(self._parent_adata, f'{annotation_key}{matrix_type}')
        
        for m_key in vdata_mm.keys():
            if m_key.startswith('__'):
                continue
            view_m = vdata_mm[m_key]
            if m_key not in parent_mm.keys():
                parent_m = None
            else:
                parent_m = parent_mm[m_key]
            
            m = self._diff_matrix(
                view_m, view_ixs, 
                parent_m, 
                parent_ixs, 
                shape=(n_rows, view_m.shape[1] if matrix_type == 'm' else n_rows),
                **kwargs
            )
            if m is not None:
                parent_mm[f'__view__{key}__{m_key}'] = m
        
    def add_view_obsm(self, vdata, key, **kwargs):
        self._assert_view_index_compatible(vdata, axis=0)
        self._add_view_m(vdata, axis=0, key=key, matrix_type='m', **kwargs)
    
    def add_view_varm(self, vdata, key, **kwargs):
        self._assert_view_index_compatible(vdata, axis=1)
        self._add_view_m(vdata, axis=1, key=key, matrix_type='m', **kwargs)
    
    def add_view_obsp(self, vdata, key, **kwargs):
        self._assert_view_index_compatible(vdata, axis=0)
        self._add_view_m(vdata, axis=0, key=key, matrix_type='p', **kwargs)
    
    def add_view_varp(self, vdata, key, **kwargs):
        self._assert_view_index_compatible(vdata, axis=1)
        self._add_view_m(vdata, axis=1, key=key, matrix_type='p', **kwargs)
    
    def _diff_matrix(
        self, 
        view_matrix, 
        view_ixs, 
        parent_matrix, 
        parent_ixs, 
        shape,
        force_full_matrix=False
    ):
        view_matrix_sub = view_matrix[view_ixs]
        
        diff = False
        sparse = scipy.sparse.issparse(view_matrix_sub)
        if parent_matrix is not None:
            parent_matrix_sub = parent_matrix[parent_ixs]
            
            if sparse and scipy.sparse.issparse(parent_matrix_sub):
                sparse = True
                if not array_equal_sparse(parent_matrix_sub, view_matrix_sub):
                    diff = True
            else:
                sparse = False
                if not np.array_equal(parent_matrix_sub, view_matrix_sub):
                    diff = True
        else:
            diff = True
        
        if diff:
            if sparse:
                # brief memory calculation
                mem = np.full((1, 1), 0, dtype=view_matrix_sub.dtype).itemsize * view_matrix_sub.shape[0] * view_matrix_sub.shape[1]
                
                if force_full_matrix or mem < 8589934592:  # > 8Gb
                    m = np.full(shape, 0, dtype=view_matrix_sub.dtype)
                    m[parent_ixs] = view_matrix_sub.toarray()
                    m = scipy.sparse.csr_matrix(m)
                else:
                    m = scipy.sparse.dok_matrix(shape, dtype=view_matrix_sub.dtype)
                    m[parent_ixs] = view_matrix_sub
                    m = m.tocsr()
            else:
                m = np.full(shape, np.nan, dtype=view_matrix_sub.dtype)
                m[parent_ixs] = view_matrix_sub

            return m
        return None
        
    def add_view_layers(self, vdata, key):
        self._assert_view_index_compatible(vdata, axis=0)
        self._assert_view_index_compatible(vdata, axis=1)
        
        parent_obs_ixs, view_obs_ixs = self._ix_converter(vdata, axis=0)
        parent_var_ixs, view_var_ixs = self._ix_converter(vdata, axis=1)
        parent_ixs = np.ix_(parent_obs_ixs, parent_var_ixs)
        view_ixs = np.ix_(view_obs_ixs, view_var_ixs)

        view_layers = vdata.layers
        parent_layers = self._parent_adata.layers
        
        for layer_key in view_layers.keys():
            if layer_key.startswith('__'):
                continue
            view_layer = view_layers[layer_key]
            
            parent_layer = None
            if layer_key in parent_layers.keys():
                parent_layer = parent_layers[layer_key]
            
            m = self._diff_matrix(view_layer, view_ixs, parent_layer, parent_ixs, shape=self._parent_adata.shape)
            if m is not None:
                parent_layers[f'__view__{key}__{layer_key}'] = m
    
    def add_view_X(self, vdata, key):
        self._assert_view_index_compatible(vdata, axis=0)
        self._assert_view_index_compatible(vdata, axis=1)
        
        parent_obs_ixs, view_obs_ixs = self._ix_converter(vdata, axis=0)
        parent_var_ixs, view_var_ixs = self._ix_converter(vdata, axis=1)
        parent_ixs = np.ix_(parent_obs_ixs, parent_var_ixs)
        view_ixs = np.ix_(view_obs_ixs, view_var_ixs)
        
        m = self._diff_matrix(vdata.X, view_ixs, self._parent_adata.X, parent_ixs, shape=self._parent_adata.shape)
        if m is not None:
            self._parent_adata.layers[f'__view__{key}__X'] = m
    
    def add_view_uns(self, vdata, key):
        view_uns = vdata.uns
        parent_uns = self._parent_adata.uns
        
        for uns_key in view_uns.keys():
            if uns_key.startswith('__'):
                continue
            
            if uns_key not in parent_uns:
                self._parent_adata.uns[f'__view__{key}__{uns_key}'] = view_uns[uns_key]
            else:
                is_diff, diff_dict = difference_dict(parent_uns, view_uns, uns_key, key)
                if not is_diff:
                    self._parent_adata.uns[uns_key] = diff_dict
                else:
                    self._parent_adata.uns[f'__view__{key}__{uns_key}'] = diff_dict
    
    def add_view_info(
        self, 
        vdata, 
        key, 
        name=None, 
        description=None, 
        group=None, 
        ignore_obs_differences=False,
        ignore_var_differences=False,
    ):
        if '__view__' not in self._parent_adata.uns.keys():
            self._parent_adata.uns['__view__'] = {}
        
        if key not in self._parent_adata.uns['__view__']:
            self._parent_adata.uns['__view__'][key] = {
                'key': key, 
                'name': key,
                'description': "",
            }
        
        # check if we need to add an index constraint for consistency
        try:
            obs_constraints = vdata.obs_constraints.copy()
        except AttributeError:
            obs_constraints = []
        
        try:
            var_constraints = vdata.var_constraints.copy()
        except AttributeError:
            var_constraints = []
        
        tmp_vdata = AnnDataView(self._parent_adata, obs_constraints, var_constraints)
        
        if not ignore_obs_differences and not np.array_equal(tmp_vdata.obs.index, vdata.obs.index):
            obs_constraints.append(IndexDataFrameConstraint(vdata.obs.index.to_numpy()))
            
        if not ignore_var_differences and not np.array_equal(tmp_vdata.var.index, vdata.var.index):
            var_constraints.append(IndexDataFrameConstraint(vdata.var.index.to_numpy()))
        
        self._parent_adata.uns['__view__'][key]['obs_constraints'] = json.dumps([c.to_dict() for c in obs_constraints])
        self._parent_adata.uns['__view__'][key]['var_constraints'] = json.dumps([c.to_dict() for c in var_constraints])

        if name is not None:
            self._parent_adata.uns['__view__'][key]['name'] = name
        
        if description is not None:
            self._parent_adata.uns['__view__'][key]['description'] = description
        
        if group is not None:
            self._parent_adata.uns['__view__'][key]['group'] = group
        
        self._parent_adata.obs[f'__view__{key}'] = self._parent_adata.obs.index.isin(vdata.obs.index)
        self._parent_adata.var[f'__view__{key}'] = self._parent_adata.var.index.isin(vdata.var.index)
    
    def view_info(self, key=None):
        info = {}
        if '__view__' not in self._parent_adata.uns.keys():
            if key is not None:
                raise ValueError("No views found in AnnData object!")
            return info
        keys = [key] if key is not None else list(self._parent_adata.uns['__view__'].keys())
        for view_key in keys:
            info[view_key] = deepcopy(self._parent_adata.uns['__view__'][view_key])
            obs_constraints = info[view_key].get('obs_constraints', None)
            if obs_constraints is not None:
                obs_constraints_dicts = json.loads(obs_constraints)
                info[view_key]['obs_constraints'] = [Constraint.from_dict(c) for c in obs_constraints_dicts]
            var_constraints = info[view_key].get('var_constraints', None)
            if var_constraints is not None:
                var_constraints_dicts = json.loads(var_constraints)
                info[view_key]['var_constraints'] = [Constraint.from_dict(c) for c in var_constraints_dicts]
            info[view_key]['valid_obs'] = self._parent_adata.obs[f'__view__{view_key}'].to_numpy()
            info[view_key]['valid_var'] = self._parent_adata.var[f'__view__{view_key}'].to_numpy()
        
        if key is not None:
            return info[key]
        return info
    
    def add_view(self, vdata, key, name=None, description=''):
        if name is None:
            name = key
        
        self.add_view_info(vdata, key, name=name, description=description)
        self.add_view_obs(vdata, key)
        self.add_view_var(vdata, key)
        self.add_view_uns(vdata, key)
        self.add_view_obsm(vdata, key)
        self.add_view_varm(vdata, key)
        self.add_view_obsp(vdata, key)
        self.add_view_varp(vdata, key)
    
    def copy(self, only_constraints=False):
        if not only_constraints:
            adata = self._parent_adata.copy()
        else:
            adata = self._parent_adata
        
        return self.__class__(
            adata, 
            obs_constraints=deepcopy(self.obs_constraints),
            var_constraints=deepcopy(self.var_constraints),
        )
    
    def drop_views(
        self,
        copy=True,
    ):
        if copy:
            adata = self._parent_adata.copy()
        else:
            adata = self._parent_adata
        
        prefix = '__view__'

        # obs
        adata.obs = adata.obs.drop(
            [
                col for col in adata.obs.columns if col.startswith(prefix)
            ],
            axis=1
        )

        # var
        adata.var = adata.var.drop(
            [
                col for col in adata.var.columns if col.startswith(prefix)
            ],
            axis=1
        )

        for slot in [
            'obsm', 'varm', 'obsp', 
            'varp', 'layers',
        ]:
            if hasattr(adata, slot):
                keys = list(getattr(adata, slot).keys())

                for key in keys:
                    if key.startswith(prefix):
                        del getattr(adata, slot)[key]

        if hasattr(adata, 'uns'):
            def remove_view_keys(d):
                d_new = dict()
                for k, v in d.items():
                    if k.startswith(prefix):
                        continue
                    if isinstance(v, dict):
                        v = remove_view_keys(v)
                    d_new[k] = v
                return d_new
            
            adata.uns = remove_view_keys(adata.uns)
        return adata


for name, (constraint_class, axis) in Constraint.plugin_methods.items():
    m = my_partialmethod(_add_constraint, axis=axis, constraint_class=constraint_class)
    m.__doc__ = constraint_class.__doc__
    m.__name__ = name
    setattr(AnnDataView, name, m)


class AnnDataSubset(AnnData):
    def __init__(self, *args, **kwargs):
        parent_adata = kwargs.pop('parent_adata', None)
        obs_constraints = kwargs.pop('obs_constraints', None)
        var_constraints = kwargs.pop('var_constraints', None)
        
        self._init(parent_adata, obs_constraints, var_constraints)

        super().__init__(*args, **kwargs)
    
    def _init(self, parent, obs_constraints=None, var_constraints=None):
        self.parent_adata = parent
        self.obs_constraints = [] if obs_constraints is None else obs_constraints
        self.var_constraints = [] if var_constraints is None else var_constraints

    def _mutated_copy(self, **kwargs):
        """Creating AnnData with attributes optionally specified via kwargs."""
        if self.isbacked:
            if "X" not in kwargs or (self.raw is not None and "raw" not in kwargs):
                raise NotImplementedError(
                    "This function does not currently handle backed objects "
                    "internally, this should be dealt with before."
                )
        new = {}

        for key in ["obs", "var", "obsm", "varm", "obsp", "varp", "layers", 
                    "parent_adata", "obs_constraints", "var_constraints"]:
            if key in kwargs:
                new[key] = kwargs[key]
            else:
                value = getattr(self, key)
                if value is not None:
                    new[key] = getattr(self, key).copy()
                else:
                    new[key] = getattr(self, key)
        if "X" in kwargs:
            new["X"] = kwargs["X"]
            new["dtype"] = new["X"].dtype
        elif self._has_X():
            new["X"] = self.X.copy()
            new["dtype"] = new["X"].dtype
        if "uns" in kwargs:
            new["uns"] = kwargs["uns"]
        else:
            new["uns"] = deepcopy(self._uns)
        if "raw" in kwargs:
            new["raw"] = kwargs["raw"]
        elif self.raw is not None:
            new["raw"] = self.raw.copy()
        return AnnDataSubset(**new)
    
    def __getitem__(self, index: Index) -> "AnnData":
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return AnnDataSubset(self, oidx=oidx, vidx=vidx, asview=True, 
                             obs_constraints=self.obs_constraints, 
                             var_constraints=self.var_constraints)
