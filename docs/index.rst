:code:`anndataview` is a Python package for working with `AnnData <https://anndata.readthedocs.io/en/latest/>`_
objects. With :code:`anndataview` you can

- Create "views" of the data by adding intuitive constraints to the observations or 
  variables of :class:`~AnnData` objects
- Save views inside the existing :code:`AnnData` object, along with any 
  view-specific data (embeddings, transformations, annotations, ...)
- Retrieve views, and the data associated with them

AnnDataView constraints
_______________________

``anndataview``'s main class, :class:`~AnnDataView`, wraps :class:`~AnnData` and acts as a proxy:

.. jupyter-execute::

   from anndata import read_h5ad
   from anndataview import AnnDataView
   
   adata = read_h5ad("data/10x_pbmc68k_reduced.h5ad")
   vdata = AnnDataView(adata)

Without further modifications, the :code:`vdata` object behaves like 
:code:`adata`: properties and methods are simply 
forwarded to the wrapped object.
 
.. jupyter-execute::

   from collections import Counter
   print(Counter(adata.obs['phase']))
   print(Counter(vdata.obs['phase']))

By adding constraints to ``vdata``, we can create a "view" of ``adata``. 
Here, we only want cells in G1 phase:

.. jupyter-execute::

   vdata.add_categorical_obs_constraint('phase', 'G1')
   print(Counter(vdata.obs['phase']))

Our constraint affects all aspects of ``adata`` - not just the ``.obs`` 
annotation table. The constraints are calculated on the fly - ``adata`` 
itself remains unmodified:

.. jupyter-execute::

   print('adata shape: ', adata.shape)
   print('vdata shape: ', vdata.shape)

Constraints can be added on observations (``obs``) and variables (``var``). 
Read more about constraints in :ref:`Constraints`.
 
.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
