import pandas as pd
try:
    from anndata.compat._overloaded_dict import OverloadedDict as DictView
except ModuleNotFoundError:
    from anndata._core.views import DictView
from anndata._core.aligned_mapping import AxisArrays, AxisArraysView, PairwiseArrays, PairwiseArraysView, LayersView, Layers
from collections.abc import Mapping
from anndata._io.specs.registry import _REGISTRY, IOSpec
from anndata._io.specs.methods import write_dataframe, write_mapping
from .utils import Proxy
import h5py
from collections.abc import MutableMapping


class DataFrameView(pd.DataFrame):
    def __call__(self, view_key=None):
        if view_key is not None:
            df = self.copy()
            column_view_prefix = f'__view__{view_key}__'
            
            columns = [c for c in df.columns]
            column_order = {name: i for i, name in enumerate(columns)}
            for i, column in enumerate(df.columns):
                if column.startswith(column_view_prefix):
                    column_new_name = column[len(column_view_prefix):]
                    try:
                        columns[column_order[column_new_name]] = f'__view__default__{column_new_name}'
                    except KeyError:
                        pass
                    columns[i] = column_new_name
            df.columns = columns
        else:
            df = self
        return df

@_REGISTRY.register_write(h5py.Group, DataFrameView, IOSpec("dataframe", "0.2.0"))
def write_dataframe_view(*args, **kwargs):
    return write_dataframe(*args, **kwargs)


def view_dict(parent, view_key):
    if isinstance(parent, Mapping):
        key_view_prefix = f'__view__{view_key}__'
        
        d = parent.copy()
        for key in parent.keys():
            if key.startswith(key_view_prefix):
                original_key = key[len(key_view_prefix):]
                if original_key in parent:
                    d[f'__view__default__{original_key}'] = view_dict(parent[original_key], view_key)
                d[original_key] = view_dict(parent[key], view_key)
                del d[key]
            else:
                d[key] = view_dict(parent[key], view_key)
        return d
    return parent


class ViewDict(MutableMapping):
    def __init__(self, wrappee):
        self._wrappee = wrappee
        
    def __getattr__(self, attr):
        return getattr(self._wrappee, attr)
    
    # The next five methods are requirements of the ABC.
    def __setitem__(self, key, value):
        self._wrappee[key] = value
    def __getitem__(self, key):
        return self._wrappee[key]
    def __delitem__(self, key):
        del self._wrappee[key]
    def __iter__(self):
        return iter(self._wrappee)
    def __len__(self):
        return len(self._wrappee)
    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        '''returns simple dict representation of the mapping'''
        return str(self._wrappee)
    def __repr__(self):
        '''echoes class, id, & reproducible representation in the REPL'''
        return repr(self._wrappee)
    
    def __call__(self, view_key=None):
        if view_key is not None:
            return view_dict(self, view_key)
        return self


@_REGISTRY.register_write(h5py.Group, ViewDict, IOSpec("dict", "0.1.0"))
def write_view_dict(*args, **kwargs):
    return write_mapping(*args, **kwargs)


def _axis_array_restore_view(original_array, view_key=None):
    if view_key is None:
        return original_array
    
    key_view_prefix = f'__view__{view_key}__'
    new_axis_array = dict()
    original_keys = set(original_array.keys())
    for key in original_keys:
        if key.startswith(key_view_prefix):
            base_key = key[len(key_view_prefix):]
            if base_key in original_keys:
                new_axis_array[f'__view__default__{base_key}'] = original_array[base_key]
            new_axis_array[base_key] = original_array[key]
        elif f'{key_view_prefix}{key}' not in original_keys:
            new_axis_array[key] = original_array[key]
    return new_axis_array


class AxisArraysCallable(AxisArrays):
    def __call__(self, view_key=None):
        return _axis_array_restore_view(self, view_key=view_key)


class AxisArraysViewCallable(AxisArraysView):
    def __call__(self, view_key=None):
        return _axis_array_restore_view(self, view_key=view_key)


class PairwiseArraysCallable(PairwiseArrays):
    def __call__(self, view_key=None):
        return _axis_array_restore_view(self, view_key=view_key)


class PairwiseArraysViewCallable(PairwiseArraysView):
    def __call__(self, view_key=None):
        return _axis_array_restore_view(self, view_key=view_key)


class LayersCallable(Layers):
    def __call__(self, view_key=None):
        return _axis_array_restore_view(self, view_key=view_key)
    
class LayersViewCallable(LayersView):
    def __call__(self, view_key=None):
        return _axis_array_restore_view(self, view_key=view_key)


callable_matcher = {
    AxisArrays: AxisArraysCallable,
    AxisArraysView: AxisArraysViewCallable,
    PairwiseArrays: PairwiseArraysCallable,
    PairwiseArraysView: PairwiseArraysViewCallable,
    Layers: LayersCallable,
    LayersView: LayersViewCallable,
}
