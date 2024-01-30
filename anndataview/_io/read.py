from anndata import read_h5ad as _read_h5ad
from .._core.anndataview import AnnDataView


def read_h5ad(*args, **kwargs) -> AnnDataView:
    """\
    Read AnnData file from .h5ad and explicitly convert to AnnDataView.
    """
    adata = _read_h5ad(*args, **kwargs)
    return AnnDataView(adata)


