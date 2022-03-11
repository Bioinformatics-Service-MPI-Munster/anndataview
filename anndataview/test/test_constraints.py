from anndataview._core.constraints import CategoricalDataFrameConstraint
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def data_frame():
    categorical_letters = ['A', 'A', 'B', 'B', 'C', 'D', 'E', 'E', 'E', 'E']
    numerical_integers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    numerical_floats = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
    
    return pd.DataFrame({
        'letters': pd.Categorical(categorical_letters),
        'integers': pd.Categorical(numerical_integers),
        'floats': pd.Categorical(numerical_floats),
    })


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
