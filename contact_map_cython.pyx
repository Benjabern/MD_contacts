import numpy as np
cimport numpy as np
from scipy.sparse import coo_matrix

def generate_final_pairs_1d_numba(np.ndarray[float, ndim=2] coords_3d,
                                  np.ndarray[float, ndim=1] coords_1d,
                                  float cutoff, int max_pairs):
    """
    For each index i, perform a 1D bounding followed by explicit 3D distance checks.
    Returns an array of valid contact pairs of shape (num_pairs, 2).
    """
    cdef int N = coords_3d.shape[0]
    cdef np.ndarray indices = np.argsort(coords_1d)
    cdef np.ndarray sorted_vals = coords_1d[indices]

    # Preallocate output array for pairs.
    cdef np.ndarray[np.int32_t, ndim=2] pairs = np.empty((max_pairs, 2), dtype=np.int32)
    cdef int count = 0
    cdef float cutoff_sq = cutoff * cutoff

    # Create memoryviews for fast C-level access.
    cdef float[:, :] coords3d = coords_3d
    cdef float[:] sorted_vals_mv = sorted_vals
    cdef np.ndarray[np.intp_t, ndim=1] indices_mv = indices  # Use np.intp_t instead of int

    cdef int i, j, k
    cdef int i_idx, neighbor
    cdef float diff0, diff1, diff2, sum_sq

    for i in range(N):
        j = i + 1
        # Expand j until the 1D difference exceeds the cutoff.
        while j < N and (sorted_vals_mv[j] - sorted_vals_mv[i]) <= cutoff:
            j += 1

        i_idx = indices_mv[i]
        # Loop over candidate neighbors.
        for k in range(i + 1, j):
            neighbor = indices_mv[k]
            diff0 = coords3d[i_idx, 0] - coords3d[neighbor, 0]
            diff1 = coords3d[i_idx, 1] - coords3d[neighbor, 1]
            diff2 = coords3d[i_idx, 2] - coords3d[neighbor, 2]
            sum_sq = diff0 * diff0 + diff1 * diff1 + diff2 * diff2
            if sum_sq < cutoff_sq:
                pairs[count, 0] = i_idx
                pairs[count, 1] = neighbor
                count += 1
                if count >= max_pairs:
                    raise RuntimeError("Ran out of space in pairs array (Cython). Increase max_pairs estimate.")
    return pairs[:count].copy()


def build_contact_map_single_axis_cython(np.ndarray coords, float cutoff, list res_indices):
    """
    Build a sparse contact map using the cythonized function for pair generation.
    """
    cdef int N = coords.shape[0]
    cdef np.ndarray stds = np.std(coords, axis=0)
    cdef int max_dim = np.argmax(stds)

    # 1D projection based on the chosen dimension.
    cdef np.ndarray coord_1d = coords[:, max_dim]
    cdef float min_val = coord_1d.min()
    cdef float max_val = coord_1d.max()
    cdef float range_1d = max_val - min_val

    cdef int max_pairs
    cdef float density_1d = 0.0
    max_pairs = (N * N)
    # Generate valid contact pairs.
    cdef np.ndarray pairs_ij = generate_final_pairs_1d_numba(coords, coord_1d, cutoff, max_pairs)

    cdef np.ndarray i_idx = pairs_ij[:, 0]
    cdef np.ndarray j_idx = pairs_ij[:, 1]

    # Create a dictionary to map atom indices to residue indices
    cdef int num_residues = len(res_indices)
    cdef int[:] atom_to_residue = np.empty(N, dtype=np.int32)
    cdef int res_i, atom_index
    for res_i in range(num_residues):
        for atom_index in res_indices[res_i]:
            atom_to_residue[atom_index] = res_i

    # Initialize the residue contact matrix
    cdef np.ndarray[np.uint32_t, ndim=2] residue_contact_matrix = np.zeros((num_residues, num_residues), dtype=np.uint32)

    # Iterate over contacts and update the residue contact matrix
    cdef int i, j, res_j
    for k in range(pairs_ij.shape[0]):
        i = pairs_ij[k, 0]
        j = pairs_ij[k, 1]
        res_i = atom_to_residue[i]
        res_j = atom_to_residue[j]
        residue_contact_matrix[res_i, res_j] = 1
        residue_contact_matrix[res_j, res_i] = 1  # Ensure symmetry

    return residue_contact_matrix