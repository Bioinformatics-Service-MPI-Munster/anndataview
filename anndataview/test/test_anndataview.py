import os
from anndata import AnnData
from anndataview import AnnDataView, read
from anndataview._core.annotations import AxisArraysCallable, AxisArraysViewCallable
import pandas as pd
import numpy as np
import scipy.sparse

import pytest


@pytest.fixture
def vdata_pbmc():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return read(os.path.join(test_dir, '10x_pbmc68k_reduced.h5ad'))


@pytest.fixture
def vdata():
    obs = pd.DataFrame(
        {
            'letter': pd.Categorical(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']),
            'score_int': np.array(range(10)),
            'score_float': np.arange(0, 1, 0.1),
        }, 
        index=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    )
    
    var = pd.DataFrame(
        {
            'letter': pd.Categorical(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',]),
            'score_int': np.array(range(9)),
            'score_float': np.arange(0, 0.9, 0.1),
        }, 
        index=['var_A', 'var_B', 'var_C', 'var_D', 'var_E', 'var_F', 'var_G', 'var_H', 'var_I'],
    )
    data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 1, 2, 3, 3, 3, 3, 3, 3, 3],
            [0, 1, 2, 3, 4, 4, 4, 4, 4, 4],
            [0, 1, 2, 3, 4, 5, 5, 5, 5, 5],
            [0, 1, 2, 3, 4, 5, 6, 6, 6, 6],
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 7],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    data_sparse = scipy.sparse.lil_matrix(np.array(data)).tocsr(copy=True)
    
    X = data_sparse[:obs.shape[0], :var.shape[0]]

    obsm = {'int': np.array(data)[:obs.shape[0], :3],
            'int_sparse': data_sparse[:obs.shape[0], :3],
            'float': np.array(data)[:obs.shape[0], :3]/2}
    varm = {'int': np.array(data).T[:var.shape[0], :3],
            'int_sparse': data_sparse.T[:var.shape[0], :3],
            'float': np.array(data).T[:var.shape[0], :3]/2}
    obsp = {'int': np.array(data)[:obs.shape[0], :obs.shape[0]],
            'int_sparse': data_sparse[:obs.shape[0], :obs.shape[0]],
            'float': np.array(data)[:obs.shape[0], :obs.shape[0]]/2}
    varp = {'int': np.array(data)[:var.shape[0], :var.shape[0]],
            'int_sparse': data_sparse[:var.shape[0], :var.shape[0]],
            'float': np.array(data)[:var.shape[0], :var.shape[0]]/2}
    
    adata = AnnData(
        X=X,
        obs=obs,
        var=var,
        layers={
            'counts': X.copy()
        },
        obsm=obsm,
        varm=varm,
        obsp=obsp,
        varp=varp,
    )
    return AnnDataView(adata)


class TestAnnDataView:
    def test_basic(self, vdata):
        assert vdata.shape[0] == 10
        assert vdata.shape[1] == 9
        
    def test_unconstrained(self, vdata):
        assert vdata.shape == (10, 9)
        assert vdata.X.shape == (10, 9)
        assert vdata.layers['counts'].shape == (10, 9)
        assert vdata.obs.shape[0] == 10
        assert vdata.obsm['int'].shape == (10, 3)
        assert vdata.obsm['float'].shape == (10, 3)
        assert vdata.varm['int'].shape == (9, 3)
        assert vdata.obsp['int'].shape == (10, 10)
        assert vdata.varp['int'].shape == (9, 9)
    
    def test_categorical_obs_constraint_single(self, vdata):
        vdata.add_categorical_obs_constraint('letter', 'C')
        
        # check data shapes and content correct
        assert vdata.shape == (1, 9)
        assert vdata.X.shape == (1, 9)
        assert np.array_equal(vdata.X.toarray(), [[0, 1, 2, 2, 2, 2, 2, 2, 2,]])
        assert vdata.layers['counts'].shape == (1, 9)
        assert np.array_equal(vdata.layers['counts'].toarray(), [[0, 1, 2, 2, 2, 2, 2, 2, 2,]])
        assert vdata.obs.shape[0] == 1
        assert vdata.obs['letter'][0] == 'C'
        assert vdata.obs['score_int'][0] == 2
        assert vdata.obs_vector('score_float')[0] == 0.2
        assert vdata.obsm['int'].shape == (1, 3)
        assert np.array_equal(vdata.obsm['int'], [[0, 1, 2,]])
        assert vdata.obsm['float'].shape == (1, 3)
        assert vdata.varm['int'].shape == (9, 3)
        assert vdata.obsp['int'].shape == (1, 1)
        assert np.array_equal(vdata.obsp['int'], [[2]])
        assert vdata.varp['int'].shape == (9, 9)
        
    def test_categorical_obs_constraint_multiple(self, vdata):
        vdata.add_categorical_obs_constraint('letter', ['A', 'C'])
        
        # check data shapes and content correct
        assert vdata.shape == (2, 9)
        assert vdata.X.shape == (2, 9)
        assert np.array_equal(vdata.X.toarray(), [[0, 0, 0, 0, 0, 0, 0, 0, 0,], [0, 1, 2, 2, 2, 2, 2, 2, 2,]])
        assert vdata.layers['counts'].shape == (2, 9)
        assert np.array_equal(vdata.layers['counts'].toarray(), [[0, 0, 0, 0, 0, 0, 0, 0, 0,], [0, 1, 2, 2, 2, 2, 2, 2, 2,]])
        assert vdata.obs.shape[0] == 2
        assert np.array_equal(vdata.obs['letter'], ['A', 'C'])
        assert np.array_equal(vdata.obs['score_int'], [0, 2])
        assert np.array_equal(vdata.obs_vector('score_float'), [0., .2])
        assert vdata.obsm['int'].shape == (2, 3)
        assert np.array_equal(vdata.obsm['int'], [[0, 0, 0,], [0, 1, 2,]])
        assert vdata.obsm['float'].shape == (2, 3)
        assert vdata.varm['int'].shape == (9, 3)
        assert vdata.obsp['int'].shape == (2, 2)
        assert np.array_equal(vdata.obsp['int'], [[0, 0], [0, 2]])
        assert vdata.varp['int'].shape == (9, 9)
        
    def test_categorical_var_constraint_multiple(self, vdata):
        vdata.add_categorical_var_constraint('letter', ['A', 'C'])
        
        # check data shapes and content correct
        assert vdata.shape == (10, 2)
        assert vdata.X.shape == (10, 2)
        assert np.array_equal(vdata.X.toarray().T, [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]])
        assert vdata.layers['counts'].shape == (10, 2)
        assert np.array_equal(vdata.layers['counts'].toarray().T, [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]])
        assert vdata.obs.shape[0] == 10
        assert vdata.var.shape[0] == 2
        assert np.array_equal(vdata.var['letter'], ['A', 'C'])
        assert np.array_equal(vdata.var['score_int'], [0, 2])
        assert np.array_equal(vdata.var_vector('score_float'), [0., .2])
        assert vdata.obsm['int'].shape == (10, 3)
        assert vdata.obsm['float'].shape == (10, 3)
        assert vdata.varm['int'].shape == (2, 3)
        assert np.array_equal(vdata.varm['int'], [[0, 0, 0,], [0, 1, 2,]])
        assert vdata.varp['int'].shape == (2, 2)
        assert np.array_equal(vdata.varp['int'].T, [[0, 0], [0, 2]])
        
    def test_view_obs_save(self, vdata):
        view = vdata.add_categorical_obs_constraint('letter', ['A', 'C']).adata_view
        
        expected = {
            'test': (pd.Categorical(['foo', 'bar']), None),
            'number': ([10, 11], None),
            'letter': (['X', 'Y'], ['A', 'C']),
        }
        for name, (view_value, original_value) in expected.items():
            view.obs[name] = view_value
        
        # save view
        vdata.add_view_obs(view, 'testview')
        
        # check new columns are there
        for name in expected.keys():
            assert f'__view__testview__{name}' in vdata.obs.columns
        
        # check view works in constrained vdata
        for name, (view_value, original_value) in expected.items():
            if original_value is None:
                assert name not in vdata.obs.columns
            else:
                assert np.array_equal(vdata.obs[name], original_value)
            assert np.array_equal(vdata.obs[f'__view__testview__{name}'], view_value)
            assert np.array_equal(vdata.obs('testview')[name], view_value)
            assert np.array_equal(vdata.obs_vector(name, view_key='testview'), view_value)
        
        # check view works in unconstrained vdata
        valid_ixs = vdata.valid_obs
        vdata.clear_constraints()
        for name, (view_value, original_value) in expected.items():
            is_numeric = False
            if pd.api.types.is_numeric_dtype(vdata.obs[f'__view__testview__{name}']):
                is_numeric = True
                original_full = [np.nan] * vdata.obs.shape[0]
                view_full = [np.nan] * vdata.obs.shape[0]
            else:
                original_full = ['NA'] * vdata.obs.shape[0]
                view_full = ['NA'] * vdata.obs.shape[0]
            
            for i, ix in enumerate(np.where(valid_ixs)[0]):
                if original_value is not None:
                    original_full[ix] = original_value[i]
                view_full[ix] = view_value[i]
            
            if original_value is None:
                assert name not in vdata.obs.columns
            assert np.array_equal(vdata.obs[f'__view__testview__{name}'], view_full, equal_nan=is_numeric)
            assert np.array_equal(vdata.obs('testview')[f'{name}'], view_full, equal_nan=is_numeric)
            assert np.array_equal(vdata.obs_vector(f'{name}', view_key='testview'), view_full, equal_nan=is_numeric)
            
    def test_view_var_save(self, vdata):
        view = vdata.add_categorical_var_constraint('letter', ['A', 'C']).adata_view
        
        expected = {
            'test': (pd.Categorical(['foo', 'bar']), None),
            'number': ([10, 11], None),
            'letter': (['X', 'Y'], ['A', 'C']),
        }
        for name, (view_value, original_value) in expected.items():
            view.var[name] = view_value
        
        # save view
        vdata.add_view_var(view, 'testview')
        
        # check new columns are there
        for name in expected.keys():
            assert f'__view__testview__{name}' in vdata.var.columns
        
        # check view works in constrainted vdata
        for name, (view_value, original_value) in expected.items():
            if original_value is None:
                assert name not in vdata.var.columns
            else:
                assert np.array_equal(vdata.var[name], original_value)
            assert np.array_equal(vdata.var[f'__view__testview__{name}'], view_value)
            assert np.array_equal(vdata.var('testview')[name], view_value)
            assert np.array_equal(vdata.var_vector(name, view_key='testview'), view_value)
        
        # check view works in unconstrained vdata
        valid_ixs = vdata.valid_var
        vdata.clear_constraints()
        for name, (view_value, original_value) in expected.items():
            is_numeric = False
            if pd.api.types.is_numeric_dtype(vdata.var[f'__view__testview__{name}']):
                is_numeric = True
                original_full = [np.nan] * vdata.var.shape[0]
                view_full = [np.nan] * vdata.var.shape[0]
            else:
                original_full = ['NA'] * vdata.var.shape[0]
                view_full = ['NA'] * vdata.var.shape[0]
            
            for i, ix in enumerate(np.where(valid_ixs)[0]):
                if original_value is not None:
                    original_full[ix] = original_value[i]
                view_full[ix] = view_value[i]
            
            if original_value is None:
                assert name not in vdata.var.columns
            assert np.array_equal(vdata.var[f'__view__testview__{name}'], view_full, equal_nan=is_numeric)
            assert np.array_equal(vdata.var('testview')[f'{name}'], view_full, equal_nan=is_numeric)
            assert np.array_equal(vdata.var_vector(f'{name}', view_key='testview'), view_full, equal_nan=is_numeric)
        
    def test_has_constraint(self, vdata):
        assert not vdata.has_obs_constraints
        assert not vdata.has_var_constraints
        assert not vdata.has_constraints
        
        vdata.add_categorical_obs_constraint('letter', 'A')
        assert vdata.has_obs_constraints
        assert not vdata.has_var_constraints
        assert vdata.has_constraints
        
        vdata.add_numerical_var_constraint('score_int', min_value=2)
        assert vdata.has_obs_constraints
        assert vdata.has_var_constraints
        assert vdata.has_constraints
        
    def test_valid(self, vdata):
        assert sum(vdata.valid_obs) == vdata._parent_adata.obs.shape[0]
        assert sum(vdata.valid_var) == vdata._parent_adata.var.shape[0]
        
        vdata.add_categorical_obs_constraint('letter', 'A')
        assert sum(vdata.valid_obs) == 1
        assert sum(vdata.valid_var) == vdata._parent_adata.var.shape[0]
        
        vdata.add_numerical_var_constraint('score_int', min_value=2)
        assert sum(vdata.valid_obs) == 1
        assert sum(vdata.valid_var) == 6
        
    def test_valid_ixs(self, vdata):
        assert np.array_equal(vdata.valid_obs_ixs, list(range(vdata._parent_adata.obs.shape[0])))
        assert np.array_equal(vdata.valid_var_ixs, list(range(vdata._parent_adata.var.shape[0])))
        
        vdata.add_categorical_obs_constraint('letter', ['A', 'C', 'I', 'J'])
        assert np.array_equal(vdata.valid_obs_ixs, [0, 2, 8, 9])
        assert np.array_equal(vdata.valid_var_ixs, list(range(vdata._parent_adata.var.shape[0])))
        vdata.add_numerical_obs_constraint('score_int', max_value=7)
        assert np.array_equal(vdata.valid_obs_ixs, [0, 2])
        assert np.array_equal(vdata.valid_var_ixs, list(range(vdata._parent_adata.var.shape[0])))
        
        vdata.add_categorical_var_constraint('letter', ['B', 'C', 'H',])
        assert np.array_equal(vdata.valid_obs_ixs, [0, 2])
        assert np.array_equal(vdata.valid_var_ixs, [1, 2, 7])
        vdata.add_numerical_var_constraint('score_int', min_value=3)
        assert np.array_equal(vdata.valid_obs_ixs, [0, 2])
        assert np.array_equal(vdata.valid_var_ixs, [7])
        
    def test_anndata_subset(self, vdata):
        vdata.add_categorical_obs_constraint('letter', ['A', 'C', 'I', 'J'])
        vdata.add_numerical_var_constraint('score_int', min_value=3)
        
        view = vdata.adata_view
        assert hasattr(view, 'parent_adata')
        assert hasattr(view, 'obs_constraints')
        assert hasattr(view, 'var_constraints')
        
        assert len(view.obs_constraints) == 1
        assert len(view.var_constraints) == 1
    
    def test_dimensions(self, vdata):
        assert vdata.n_obs == 10
        assert vdata.n_vars == 9
        
        vdata.add_categorical_obs_constraint('letter', ['A', 'C', 'I', 'J'])
        vdata.add_numerical_var_constraint('score_int', min_value=3)
        
        assert vdata.n_obs == 4
        assert vdata.n_vars == 5
    
    def test_repr(self, vdata):
        vdata.add_categorical_obs_constraint('letter', 'A')
        vdata.add_categorical_var_constraint('letter', 'A')
        r = repr(vdata)
        assert 'AnnDataView' in r
        assert 'obs constraints (1):' in r
        assert 'var constraints (1):' in r

    def test_index_compatible(self, vdata):
        view = (vdata.add_categorical_obs_constraint('letter', ['A', 'C'])
                     .add_categorical_var_constraint('letter', ['B', 'D'])
                     .adata_view)
        
        vdata._assert_view_index_compatible(view, 0)
        vdata._assert_view_index_compatible(view, 1)
        
        with pytest.raises(ValueError):
            vdata._assert_view_index_compatible(view, 2)
        
        view.obs.index = ['foo', 'bar']
        view.var.index = ['foo', 'bar']
        
        with pytest.raises(ValueError):
            vdata._assert_view_index_compatible(view, 0)
        with pytest.raises(ValueError):
            vdata._assert_view_index_compatible(view, 1)
    
    def test_view_obsm_save(self, vdata):
        view = vdata.add_categorical_obs_constraint('letter', ['A', 'C']).adata_view
        
        # modification
        view.obsm['int'] = vdata.obsm['int'] * 2
        view.obsm['int_sparse'] = scipy.sparse.lil_matrix(vdata.obsm['int'] * 2).tocsr()
        # addition
        view.obsm['test'] = np.full((2, 2), 1)
        
        vdata.add_view_obsm(view, 'testview')
        
        assert np.array_equal(vdata.obsm['__view__testview__test'], [[1, 1], [1, 1]])
        assert np.array_equal(vdata.obsm['__view__testview__int'], [[0, 0, 0], [0, 2, 4]])
        assert np.array_equal(vdata.obsm['__view__testview__int_sparse'].toarray(), [[0, 0, 0], [0, 2, 4]])
        
        vdata.clear_constraints()
        expected_int_full = np.full(vdata.obsm['int'].shape, np.nan)
        expected_int_full[0] = [0, 0, 0]
        expected_int_full[2] = [0, 2, 4]
        expected_int_sparse_full = np.full(vdata.obsm['int'].shape, 0)
        expected_int_sparse_full[0] = [0, 0, 0]
        expected_int_sparse_full[2] = [0, 2, 4]
        expected_test_full = np.full((vdata.shape[0], 2), np.nan)
        expected_test_full[0] = [1, 1]
        expected_test_full[2] = [1, 1]
        assert np.array_equal(vdata.obsm['__view__testview__test'], expected_test_full, equal_nan=True)
        assert np.array_equal(vdata.obsm['__view__testview__int'], expected_int_full, equal_nan=True)
        assert np.array_equal(vdata.obsm['__view__testview__int_sparse'].toarray(), expected_int_sparse_full, equal_nan=True)
        
        assert isinstance(vdata.obsm, (AxisArraysViewCallable, AxisArraysCallable))
        restored_obsm = vdata.obsm(view_key='testview')
        print(restored_obsm)
        assert np.array_equal(restored_obsm['test'], expected_test_full, equal_nan=True)
        assert np.array_equal(restored_obsm['int'], expected_int_full, equal_nan=True)
        assert np.array_equal(restored_obsm['int_sparse'].toarray(), expected_int_sparse_full, equal_nan=True)
        
    def test_view_varm_save(self, vdata):
        view = vdata.add_categorical_var_constraint('letter', ['A', 'C']).adata_view
        
        # modification
        view.varm['int'] = vdata.varm['int'] * 2
        # addition
        view.varm['test'] = np.full((2, 2), 1)
        
        vdata.add_view_varm(view, 'testview')
        
        assert np.array_equal(vdata.varm['__view__testview__test'], [[1, 1], [1, 1]])
        assert np.array_equal(vdata.varm['__view__testview__int'], [[0, 0, 0], [0, 2, 4]])
        
        vdata.clear_constraints()
        expected_int_full = np.full(vdata.varm['int'].shape, np.nan)
        expected_int_full[0] = [0, 0, 0]
        expected_int_full[2] = [0, 2, 4]
        expected_test_full = np.full((vdata.shape[1], 2), np.nan)
        expected_test_full[0] = [1, 1]
        expected_test_full[2] = [1, 1]
        assert np.array_equal(vdata.varm['__view__testview__test'], expected_test_full, equal_nan=True)
        assert np.array_equal(vdata.varm['__view__testview__int'], expected_int_full, equal_nan=True)
        
    def test_view_obsp_save(self, vdata):
        view = vdata.add_categorical_obs_constraint('letter', ['A', 'C']).adata_view
        
        # modification
        view.obsp['int'] = vdata.obsp['int'] * 2
        # addition
        view.obsp['test'] = np.full((2, 2), 1)
        # sparse
        view.obsp['int_sparse'] = vdata.obsp['int_sparse'] * 2
        print('norm', view.obsp['int'])
        print('sparse', view.obsp['int_sparse'])
        
        vdata.add_view_obsp(view, 'testview')
        
        assert np.array_equal(vdata.obsp['__view__testview__test'], [[1, 1], [1, 1]])
        assert np.array_equal(vdata.obsp['__view__testview__int'], [[0, 0], [0, 4]])
        assert np.array_equal(vdata.obsp['__view__testview__int_sparse'].toarray(), [[0, 0], [0, 4]])
        
        vdata.clear_constraints()
        expected_int_full = np.full(vdata.obsp['int'].shape, np.nan)
        expected_int_full[0, 0] = 0
        expected_int_full[0, 2] = 0
        expected_int_full[2, 0] = 0
        expected_int_full[2, 2] = 4
        expected_test_full = np.full((vdata.shape[0], vdata.shape[0]), np.nan)
        expected_test_full[0, 0] = 1
        expected_test_full[0, 2] = 1
        expected_test_full[2, 0] = 1
        expected_test_full[2, 2] = 1
        assert np.array_equal(vdata.obsp['__view__testview__test'], expected_test_full, equal_nan=True)
        assert np.array_equal(vdata.obsp['__view__testview__int'], expected_int_full, equal_nan=True)
        
        assert '__view__testview__float' not in vdata.obsp.keys()

    def test_view_varp_save(self, vdata):
        view = vdata.add_categorical_var_constraint('letter', ['A', 'C']).adata_view
        
        # modification
        view.varp['int'] = vdata.varp['int'] * 2
        # addition
        view.varp['test'] = np.full((2, 2), 1)
        
        vdata.add_view_varp(view, 'testview')
        
        print(vdata.varp['int'])
        
        assert np.array_equal(vdata.varp['__view__testview__test'], [[1, 1], [1, 1]])
        assert np.array_equal(vdata.varp['__view__testview__int'], [[0, 0], [0, 4]])
        
        vdata.clear_constraints()
        expected_int_full = np.full(vdata.varp['int'].shape, np.nan)
        expected_int_full[0, 0] = 0
        expected_int_full[0, 2] = 0
        expected_int_full[2, 0] = 0
        expected_int_full[2, 2] = 4
        expected_test_full = np.full((vdata.shape[1], vdata.shape[1]), np.nan)
        expected_test_full[0, 0] = 1
        expected_test_full[0, 2] = 1
        expected_test_full[2, 0] = 1
        expected_test_full[2, 2] = 1
        assert np.array_equal(vdata.varp['__view__testview__test'], expected_test_full, equal_nan=True)
        assert np.array_equal(vdata.varp['__view__testview__int'], expected_int_full, equal_nan=True)
        
        assert '__view__testview__float' not in vdata.varp.keys()

    def test_view_layers_save(self, vdata):
        view = (vdata.add_categorical_obs_constraint('letter', ['A', 'C'])
                     .add_categorical_var_constraint('letter', ['A', 'C'])
                     .adata_view)
        
        # modifiy
        view.layers['counts'] = scipy.sparse.lil_matrix([[1, 2], [3, 4]]).tocsr()
        # add
        view.layers['test_sparse'] = scipy.sparse.lil_matrix([[2, 4], [4, 8]]).tocsr()
        view.layers['test_full'] = np.array([[2, 4], [4, 8]])
        
        vdata.add_view_layers(view, 'testview')
        
        assert np.array_equal(vdata.layers['__view__testview__counts'].toarray(), [[1, 2], [3, 4]])
        assert np.array_equal(vdata.layers['__view__testview__test_sparse'].toarray(), [[2, 4], [4, 8]])
        assert np.array_equal(vdata.layers['__view__testview__test_full'], [[2, 4], [4, 8]])

        vdata.clear_constraints()
        expected_full = np.full(vdata.layers['counts'].shape, np.nan)
        expected_full[0, 0] = 2
        expected_full[0, 2] = 4
        expected_full[2, 0] = 4
        expected_full[2, 2] = 8
        assert np.array_equal(vdata.layers['__view__testview__test_full'], expected_full, equal_nan=True)
        expected_sparse = np.full(vdata.layers['counts'].shape, 0)
        expected_sparse[0, 0] = 2
        expected_sparse[0, 2] = 4
        expected_sparse[2, 0] = 4
        expected_sparse[2, 2] = 8
        assert np.array_equal(vdata.layers['__view__testview__test_sparse'].toarray(), expected_sparse, equal_nan=True)
        expected_counts = np.full(vdata.layers['counts'].shape, 0)
        expected_counts[0, 0] = 1
        expected_counts[0, 2] = 2
        expected_counts[2, 0] = 3
        expected_counts[2, 2] = 4
        assert np.array_equal(vdata.layers['__view__testview__counts'].toarray(), expected_counts, equal_nan=True)
        
    def test_view_X_save(self, vdata):
        view = (vdata.add_categorical_obs_constraint('letter', ['A', 'C'])
                     .add_categorical_var_constraint('letter', ['A', 'C'])
                     .adata_view)
        
        # modifiy
        view.X = scipy.sparse.lil_matrix([[1, 2], [3, 4]]).tocsr()
        
        vdata.add_view_X(view, 'testview')
        
        assert np.array_equal(vdata.layers['__view__testview__X'].toarray(), [[1, 2], [3, 4]])
        vdata.clear_constraints()
        expected_full = np.full(vdata.layers['__view__testview__X'].shape, 0)
        expected_full[0, 0] = 1
        expected_full[0, 2] = 2
        expected_full[2, 0] = 3
        expected_full[2, 2] = 4
        assert np.array_equal(vdata.layers['__view__testview__X'].toarray(), expected_full, equal_nan=True)
    
    def test_view_uns_save(self, vdata, tmp_path):
        vdata._parent_adata.uns['dict'] = {'a': 0, 'b': 'foo'}
        vdata._parent_adata.uns['dict2'] = {'a': {'x': pd.DataFrame({'x': [1,2], 'y': [3,4]})}, 'b': np.array([1,2])}
        vdata._parent_adata.uns['matrix'] = np.array([[0, 1], [2, 3]])
        vdata._parent_adata.uns['df'] = pd.DataFrame({'a': [0, 1, 2, 3], 'b': ['foo', 'bar', 'baz', 'bam']})
        
        d = tmp_path / "sub"
        d.mkdir()
        vdata.write(d / "vdata.h5ad")
        
        vdata = read(d / "vdata.h5ad")
        
        view = vdata.adata_view
        
        view.uns['dict']['a'] = 1
        view.uns['matrix'] = np.array([])
        view.uns['df'].loc[:, 'b'] = ['aaa', 'bbb', 'ccc', 'ddd']
        view.uns['dict2']['a']['x'].loc[:, 'y'] = 'z'
        view.uns['__foo__'] = 0
        view.uns['new'] = 'new'
        
        vdata.add_view_uns(view, 'testview')
        
        assert '__view__testview__a' in vdata._parent_adata.uns['dict']
        assert '__view__testview__new' in vdata._parent_adata.uns
        assert '__view__testview__b' not in vdata._parent_adata.uns['dict']
        assert '__view__testview__b' not in vdata._parent_adata.uns['dict2']
        assert '__view__testview__matrix' in vdata._parent_adata.uns_keys()
        assert '__view__testview__df' in vdata._parent_adata.uns_keys()
        assert '__view__testview__x' in vdata._parent_adata.uns['dict2']['a']
        assert '__view__testview__y' not in vdata._parent_adata.uns['dict2']['a']
        assert '__view__testview____foo__' not in vdata._parent_adata.uns
        
        view_uns = vdata.uns('testview')
        assert '__view__testview__a' not in view_uns['dict']
        assert '__view__default__a' in view_uns['dict']
        assert '__view__default__b' not in view_uns['dict']
        assert '__view__default__b' not in view_uns['dict2']
        assert '__view__default__matrix' in view_uns.keys()
        assert '__view__default__df' in view_uns.keys()
        assert '__view__default__x' in view_uns['dict2']['a']
        assert '__view__default__y' not in view_uns['dict2']['a']
        assert '__foo__' not in view_uns
        assert 'new' in view_uns
        
    def test_view_info_save(self, vdata):
        view = (vdata.add_categorical_obs_constraint('letter', ['A', 'C'])
              .add_categorical_var_constraint('letter', ['B', 'D'])
              .add_numerical_obs_constraint('score_int', max_value=1, exclusive=True)
              .add_numerical_var_constraint('score_int', min_value=1, exclusive=False)
              .adata_view
        )
        vdata.add_view_info(view, key='testview', name='Test view', 
                            description='Some long description with special characters $\n/_`ä')

        view_info = vdata.view_info('testview')
        assert view_info['key'] == 'testview'
        assert view_info['name'] == 'Test view'
        assert view_info['description'] == 'Some long description with special characters $\n/_`ä'
        assert len(view_info['obs_constraints']) == 2
        assert len(view_info['var_constraints']) == 2
        assert np.array_equal(view_info['valid_obs'], [True, False, False, False, False, False, False, False, False, False])
        assert np.array_equal(view_info['valid_var'], [False, True, False, True, False, False, False, False, False])
        
        assert '__view__' in vdata._parent_adata.uns_keys()
        assert 'testview' in vdata._parent_adata.uns['__view__']
        
    def test_view_info_save_with_additional_constraints(self, vdata):
        view = (vdata.add_categorical_obs_constraint('letter', ['A', 'B', 'C'])
              .add_categorical_var_constraint('letter', ['A', 'B', 'D'])
              .adata_view
        )
        view_sub = view[[0, 2], :]
        view_sub = view_sub[:, [0, 1]]
        
        vdata.add_view_info(view_sub, key='testview', name='Test view', 
                            description='Some long description with special characters $\n/_`ä')
        
        view_info = vdata.view_info('testview')
        assert view_info['key'] == 'testview'
        assert view_info['name'] == 'Test view'
        assert view_info['description'] == 'Some long description with special characters $\n/_`ä'
        assert len(view_info['obs_constraints']) == 2
        assert len(view_info['var_constraints']) == 2
        assert np.array_equal(view_info['valid_obs'], [True, False, True, False, False, False, False, False, False, False])
        assert np.array_equal(view_info['valid_var'], [True, True, False, False, False, False, False, False, False])