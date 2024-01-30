from anndataview._core.constraints import CategoricalDataFrameConstraint, NumericalConstraint, IndexDataFrameConstraint
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def data_frame():
    categorical_letters = ['A', 'A', 'B', 'B', 'C', 'D', 'E', 'E', 'E', 'E']
    numerical_integers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,]
    numerical_floats = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9,]
    index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    
    return pd.DataFrame({
        'letters': pd.Categorical(categorical_letters),
        'integers': numerical_integers,
        'floats': numerical_floats,
    }, index=index)


@pytest.fixture
def categorical_constraint():
    return CategoricalDataFrameConstraint(key='letters', categories=['A', 'E'])


class TestCategoricalDataFrameConstraint:
    def test_from_dict(self):
        constraint_dict = {
            'key': 'letters',
            'categories': ['A', 'E'],
        }
        constraint = CategoricalDataFrameConstraint.from_dict(constraint_dict)
        
        assert constraint.key == 'letters'
        assert constraint.categories is not None
        for i, c in enumerate(constraint.categories):
            assert c == constraint_dict['categories'][i]

    def test_to_dict(self, categorical_constraint):
        constraint_dict = categorical_constraint.to_dict()
        
        assert constraint_dict['key'] == 'letters'
        for i, c in enumerate(categorical_constraint.categories):
            assert c == constraint_dict['categories'][i]
        assert constraint_dict['name'] == 'CategoricalDataFrameConstraint'
        assert constraint_dict['class'] == 'anndataview._core.constraints.CategoricalDataFrameConstraint'
        
    def test_valid(self, data_frame, categorical_constraint):
        valid = categorical_constraint.valid(data_frame)
        assert len(valid) == data_frame.shape[0]
        assert sum(valid) == 6
        
        assert np.array_equal(valid, [True, True, False, False, False, False, True, True, True, True])
    
    def test_filter(self, data_frame, categorical_constraint):
        data_frame_sub = categorical_constraint.filter(data_frame)
        assert data_frame_sub.shape[0] == 6
        assert np.array_equal(data_frame_sub['letters'], ['A', 'A', 'E', 'E', 'E', 'E'])


@pytest.fixture
def numerical_constraint():
    return NumericalConstraint(key='floats', min_value=0.1, max_value=0.7, exclusive=False)

@pytest.fixture
def numerical_constraint_dict():
    return {
        'key': 'floats',
        'max_value': 0.7,
        'min_value': 0.1,
        'exclusive': False,
    }


class TestNumericalDataFrameConstraint:
    def test_from_dict(self, numerical_constraint_dict):
        constraint = NumericalConstraint.from_dict(numerical_constraint_dict)
        
        assert constraint.key == 'floats'
        assert constraint.min_value == 0.1
        assert constraint.max_value == 0.7
        assert constraint.exclusive == False

    def test_to_dict(self, numerical_constraint):
        constraint_dict = numerical_constraint.to_dict()
        
        assert constraint_dict['key'] == 'floats'
        assert constraint_dict['min_value'] == 0.1
        assert constraint_dict['max_value'] == 0.7
        assert constraint_dict['exclusive'] == False
        
    def test_valid(self, data_frame, numerical_constraint):
        valid = numerical_constraint.valid(data_frame)
        assert len(valid) == data_frame.shape[0]
        assert sum(valid) == 7
        
        assert np.array_equal(valid, [False, True, True, True, True, True, True, True, False, False])
    
    def test_filter(self, data_frame, numerical_constraint):
        data_frame_sub = numerical_constraint.filter(data_frame)
        assert data_frame_sub.shape[0] == 7
        assert np.array_equal(data_frame_sub['floats'], [.1, .2, .3, .4, .5, .6, .7])


@pytest.fixture
def index_constraint():
    return IndexDataFrameConstraint(indices=['a', 'b', 'e', 'h'])

@pytest.fixture
def index_constraint_dict():
    return {
        'indices': ['a', 'b', 'e', 'h']
    }


class TestIndexDataFrameConstraint:
    def test_from_dict(self, index_constraint_dict):
        constraint = IndexDataFrameConstraint.from_dict(index_constraint_dict)
        assert np.array_equal(constraint.indices, index_constraint_dict['indices'])

    def test_to_dict(self, index_constraint):
        constraint_dict = index_constraint.to_dict()
        assert np.array_equal(index_constraint.indices, constraint_dict['indices'])
        
    def test_valid(self, data_frame, index_constraint):
        valid = index_constraint.valid(data_frame)
        assert len(valid) == data_frame.shape[0]
        assert sum(valid) == 4
        assert np.array_equal(valid, [True, True, False, False, True, False, False, True, False, False])
    
    def test_filter(self, data_frame, index_constraint):
        data_frame_sub = index_constraint.filter(data_frame)
        assert data_frame_sub.shape[0] == 4
        assert np.array_equal(data_frame_sub.index, ['a', 'b', 'e', 'h'])
