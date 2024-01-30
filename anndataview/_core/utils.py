import numpy as np
import pandas as pd
import functools
from deepdiff import DeepDiff
from collections.abc import Mapping
import scipy.sparse


def array_equal_sparse(A, B, atol = 1e-8):
    """
    https://stackoverflow.com/a/47771340
    Answer by Divakar
    """
    # check matrix shape
    if np.array_equal(A.shape, B.shape)==0:
        return False

    r1,c1 = A.nonzero()
    r2,c2 = B.nonzero()

    lidx1 = np.ravel_multi_index((r1,c1), A.shape)
    lidx2 = np.ravel_multi_index((r2,c2), B.shape)

    sidx1 = lidx1.argsort()
    sidx2 = lidx2.argsort()

    index_match = np.array_equal(lidx1[sidx1], lidx2[sidx2])
    if index_match==0:
        return False
    else:  
        v1 = A.data
        v2 = B.data        
        V1 = v1[sidx1]
        V2 = v2[sidx2]        
        return np.allclose(V1,V2, atol=atol)


def my_partialmethod(func, *args1, **kwargs1):
    """
    https://stackoverflow.com/a/32997046
    Answer by Blckknght
    """
    @functools.wraps(func)  # copy attributes to start, they can be overwritten later
    def method(self, *args2, **kwargs2):
        return func(self, *args1, *args2, **kwargs1, **kwargs2)
    return method


def recursive_find_data_frames(dict_like, current_path=None):
    change_keys = []
    current_path = current_path or []
    
    try:
        keys = list(dict_like.keys())
    except AttributeError:
        return change_keys
    
    for key in keys:
        if isinstance(dict_like[key], pd.DataFrame):
            change_keys.append(tuple(current_path + [key]))
        else:
            change_keys += recursive_find_data_frames(dict_like[key], current_path + [key])
    
    return change_keys


def difference_data_frame(parent_df, view_df):
    # compare general shape
    if parent_df.shape != view_df.shape:
        return True, view_df
    
    # compare index
    if not parent_df.index.astype('object').equals(view_df.index.astype('object')):
        return True, view_df
    
    # compare columns
    for parent_column, view_column in zip(parent_df.columns, view_df.columns):
        if parent_column != view_column:
            return True, view_df
    
    # compare and content
    for column in view_df.columns:
        if (column not in parent_df or 
            not parent_df[column].astype('object')
                .equals(view_df[column].astype('object'))):
            return True, view_df
    
    return False, parent_df


def difference_dict(parent, view, uns_key, view_key):
    if uns_key not in parent:
        return True, view[uns_key]

    if isinstance(parent[uns_key], Mapping) and isinstance(view[uns_key], Mapping):
        final_result = {}
        for sub_uns_key in view[uns_key].keys():
            print(uns_key, sub_uns_key)
            try:
                is_different, sub_dict = difference_dict(parent[uns_key], view[uns_key], 
                                                        sub_uns_key, view_key)
            except RecursionError:
                print(view[uns_key])
                print(parent[uns_key])
                raise
            
            if is_different:
                final_result[f'__view__{view_key}__{sub_uns_key}'] = sub_dict
                if sub_uns_key in parent[uns_key]:
                    final_result[sub_uns_key] = parent[uns_key][sub_uns_key]
            else:
                final_result[sub_uns_key] = sub_dict
            
        return False, final_result
    
    # TODO check data frame differences
    if isinstance(parent[uns_key], pd.DataFrame) and isinstance(view[uns_key], pd.DataFrame):
        diff, diff_df =  difference_data_frame(parent[uns_key], view[uns_key])
        return diff, diff_df
    elif scipy.sparse.issparse(parent[uns_key]) and scipy.sparse.issparse(view[uns_key]):
        if parent[uns_key].shape != view[uns_key].shape:
            return True, view[uns_key]
        if parent[uns_key].dtype != view[uns_key].dtype:
            return True, view[uns_key]
        if not np.array_equal(parent[uns_key].data, view[uns_key].data):
            return True, view[uns_key]
        if not np.array_equal(parent[uns_key].indptr, view[uns_key].indptr):
            return True, view[uns_key]
        if not np.array_equal(parent[uns_key].indices, view[uns_key].indices):
            return True, view[uns_key]
    else:
        diff_dict = DeepDiff(parent[uns_key], view[uns_key])
        if len(diff_dict) > 0:
            return True, view[uns_key]
    
    return False, parent[uns_key]


class Proxy(object):
    __slots__ = ["_obj", "__weakref__"]
    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)
    
    #
    # proxying (special cases)
    #
    def __getattribute__(self, name):
        return getattr(object.__getattribute__(self, "_obj"), name)
    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_obj"), name)
    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)
    
    def __nonzero__(self):
        return bool(object.__getattribute__(self, "_obj"))
    def __str__(self):
        return str(object.__getattribute__(self, "_obj"))
    def __repr__(self):
        return repr(object.__getattribute__(self, "_obj"))
    
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
        '__repr__', '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__', 
        '__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', 
        '__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__', 
        '__truediv__', '__xor__', 'next',
    ]
    
    @classmethod
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class"""
        
        def make_method(name):
            def method(self, *args, **kw):
                try:
                    return getattr(object.__getattribute__(self, "_obj"), name)(*args, **kw)
                except AttributeError:
                    print(name)
                    return getattr(self, name)
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