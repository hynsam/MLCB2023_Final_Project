def sample_prior(batch, sigma, harmonic=True):
    if harmonic:
        bid = batch['ligand'].batch
        sde = DiffusionSDE(batch.protein_sigma * sigma)

        edges = batch['ligand', 'bond_edge', 'ligand'].edge_index
        edges = edges[:, edges[0] < edges[1]]  # de-duplicate
        try:
            D, P = HarmonicSDE.diagonalize(batch['ligand'].num_nodes, edges=edges.T, lamb=sde.lamb[bid], ptr=batch['ligand'].ptr)
        except Exception as e:
            print('batch["ligand"].num_nodes', batch['ligand'].num_nodes)
            print("batch['ligand'].size", batch['ligand'].size)
            print("batch['protein'].size", batch['protein'].batch.bincount())
            print(batch.pdb_id)
            raise e
        noise = torch.randn_like(batch["ligand"].pos)

        # Integrate pairwise distance constraints
        for i in range(batch["ligand"].pos.shape[0]):
            for j in range(i + 1, batch["ligand"].pos.shape[0]):
                if abs(pairwise_distances[i, j] - stdev_threshold) > some_tolerance_value:
                    # Adjust the positions based on your constraints
                    adjustment_factor = calculate_adjustment(batch["ligand"].pos[i], batch["ligand"].pos[j])  # Placeholder function
                    batch["ligand"].pos[i] += adjustment_factor
                    batch["ligand"].pos[j] += adjustment_factor

        prior = P @ (noise / torch.sqrt(D)[:, None])
        return prior
    else:
        prior = torch.randn_like(batch["ligand"].pos)
        return prior * sigma