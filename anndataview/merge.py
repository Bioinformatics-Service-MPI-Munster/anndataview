import anndata
from ._core.anndataview import AnnDataView


def integrate_views(
    adata_main: anndata.AnnData,
    adata_sub: anndata.AnnData,
    view_key_base: str,
    view_name_base: str,
    view_description_base: str,
    view_group_base: str,
    obs_constraints: list | None = None,
    var_constraints: list | None = None,
    ignore_var_differences: bool = True,
):
    adata_sub = adata_sub.copy()

    for constraints_name, constraints in [
        ('obs_constraints', obs_constraints),
        ('var_constraints', var_constraints),
    ]:
        if constraints is not None:
            if (
                hasattr(adata_sub, constraints_name) and 
                isinstance(getattr(adata_sub, constraints_name), list)
            ):
                raise ValueError(
                    f'adata_sub already has {constraints_name}.'
                )
            
            if isinstance(constraints, list):
                setattr(adata_sub, constraints_name, constraints)
            elif hasattr(constraints, constraints_name):
                setattr(
                    adata_sub, constraints_name, 
                    getattr(adata_sub, constraints_name)
                )


    # Add original views back
    vdata = AnnDataView(adata_main)
    vdata.add_view_info(
        adata_sub, 
        view_key_base, 
        view_name_base, 
        view_description_base,
        group=view_group_base,
        ignore_var_differences=ignore_var_differences,
    )

    vdata.add_view_uns(adata_sub, view_key_base)

    vdata.add_view_obs(adata_sub, view_key_base)
    vdata.add_view_obsm(adata_sub, view_key_base)
    vdata.add_view_obsp(adata_sub, view_key_base, force_full_matrix=True)

    if not ignore_var_differences:
        vdata.add_view_var(adata_sub, view_key_base)
        vdata.add_view_varm(adata_sub, view_key_base)
        vdata.add_view_varp(adata_sub, view_key_base, force_full_matrix=True)

    processed_views = set()
    while len(processed_views) != len(adata_sub.uns['__view__']):
        n_processed = len(processed_views)
        for original_view_key in adata_sub.uns['__view__'].keys():
            
            if original_view_key in processed_views:
                continue
            
            adata_view = AnnDataView(adata_sub).restore_view(original_view_key)
            
            # fix obs constraints
            for obs_constraint in adata_view.obs_constraints:
                if obs_constraint.key.startswith('__view__'):
                    obs_constraint.key = obs_constraint.key.replace(
                        '__view__', 
                        f'__view__{view_key_base}_'
                    )
                else:
                    obs_constraint.key = f'__view__{view_key_base}__{obs_constraint.key}'
            
            adata_view.obs_constraints += adata_sub.obs_constraints
            
            view_key = f'{view_key_base}_' + original_view_key
            view_name = f'{view_name_base} ' + adata_sub.uns['__view__'][original_view_key]['name']
            view_description = f'{view_description_base}:' + adata_sub.uns['__view__'][original_view_key]['description']
            view_group = view_group_base
            if 'group' in adata_sub.uns['__view__'][original_view_key]:
                view_group += ' ' + adata_sub.uns['__view__'][original_view_key]['group']
            
            vdata = AnnDataView(adata_main)
            try:
                vdata.add_view_info(
                    adata_view, 
                    view_key, 
                    view_name, 
                    view_description,
                    group=view_group,
                    ignore_var_differences=ignore_var_differences,
                )
            except KeyError:
                continue
            
            vdata.add_view_uns(adata_view, view_key)

            vdata.add_view_obs(adata_view, view_key)
            vdata.add_view_obsm(adata_view, view_key)
            vdata.add_view_obsp(adata_view, view_key)

            if not ignore_var_differences:
                vdata.add_view_var(adata_view, view_key_base)
                vdata.add_view_varm(adata_view, view_key_base)
                vdata.add_view_varp(adata_view, view_key_base, force_full_matrix=True)

            processed_views.add(original_view_key)

        if len(processed_views) == n_processed:
            break

    return adata_main
