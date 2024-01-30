import os
import pytest

from anndataview import read


@pytest.fixture
def anndata_file_name():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(test_dir, '10x_pbmc68k_reduced.h5ad')


class TestRead:
    def test_read(self, anndata_file_name):
        vdata = read(anndata_file_name)
        
        getattr(vdata, 'obs_constraints')
        getattr(vdata, 'var_constraints')
        
        getattr(vdata, 'add_obs_constraint')
        getattr(vdata, 'add_var_constraint')
