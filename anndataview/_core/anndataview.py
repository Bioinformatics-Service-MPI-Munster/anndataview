from anndata import AnnData

class AnnDataView(AnnData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init()
    
    def _init(self):
        self.constraints = []
        self.view_name = []
    
