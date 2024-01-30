import numpy as np
import pandas as pd
from anndataview._core.utils import recursive_find_data_frames, difference_dict
from deepdiff import DeepDiff, extract
from copy import deepcopy
import pprint

class TestUtils:
    def test_df_find(self):
        d = {
            'a': pd.DataFrame({}),
            'b': {
                'w': pd.DataFrame({}),
                'x': pd.DataFrame({}),
                'y': 1,
                'z': {
                    'foo': pd.DataFrame({}),
                    'bar': np.array([]),
                },
            },
            'c': 0,
        }
        dc = deepcopy(d)
        dc['b']['z']['baz'] = 0
        dc['b']['z']['bar'] = 0

        keys = recursive_find_data_frames(d)
        assert ('a',) in keys
        assert ('b', 'w') in keys
        assert ('b', 'x') in keys
        assert ('b', 'z', 'foo') in keys
        assert len(keys) == 4
        
        diff = DeepDiff(d, dc).get('dictionary_item_added')
        
    def test_difference_dict(self):
        d = {'root': {
                'a': 'test',
                'b': {
                    'w': 0,
                    'x': 1,
                    'y': 2,
                    'z': {
                        'foo': 'foo',
                        'bar': np.array([]),
                    },
                },
                'c': 0,
            }
        }
        dc = deepcopy(d)
        dc['root']['b']['z']['baz'] = 0
        dc['root']['b']['z']['bar'] = 0
        dc['root']['d'] = {'bla': 1}

        diff, final_dict = difference_dict(d, dc, 'root', 'test')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(final_dict)
        
    def test_difference_dict_simple(self):
        d = {'root': {
                'a': 0,
                'b': {
                    'y': 0,
                    'z': {
                        'foo': 'foo',
                    },
                },
            }
        }
        dc = deepcopy(d)
        dc['root']['b']['z']['foo'] = 0

        diff, final_dict = difference_dict(d, dc, 'root', 'test')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(final_dict)
