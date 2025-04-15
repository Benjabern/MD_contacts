import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from scipy.sparse import coo_matrix
import cython

@cython.nogil
cpdef inline np.ndarray[np.int32_t, ndim=2] filter_pairs_1d(
    float[:, :] coords_3d,
    float[:] coords_1d,
    int[:] indices,
    float cutoff,
    int max_pairs
):
    cdef int N = coords_3d.shape[0]
    cdef np.ndarray[np.int32_t, ndim=2] pairs = np.empty((max_pairs, 2), dtype=np.int32)
    cdef int count = 0
    cdef float cutoff_sq = cutoff * cutoff
    cdef int i, j, k
    cdef int i_idx, neighbor
    cdef float diff0, diff1, diff2, sum_sq

    j = 1
    for i in range(N):
        while j < N and (coords_1d[j] - coords_1d[i]) <= cutoff:
            j += 1
        i_idx = indices[i]
        for k in range(i + 1, j):
            neighbor = indices[k]
            diff0 = coords_3d[i_idx, 0] - coords_3d[neighbor, 0]
            diff1 = coords_3d[i_idx, 1] - coords_3d[neighbor, 1]
            diff2 = coords_3d[i_idx, 2] - coords_3d[neighbor, 2]
            sum_sq = diff0 * diff0 + diff1 * diff1 + diff2 * diff2
            if sum_sq < cutoff_sq:
                pairs[count, 0] = i_idx
                pairs[count, 1] = neighbor
                count += 1
                if count >= max_pairs:
                    raise RuntimeError("Ran out of space in pairs array. Increase max_pairs.")
    return pairs[:count]

@cython.nogil
cpdef np.ndarray[np.float32_t, ndim=2] extend_orthorhombic(
    np.ndarray[np.float32_t, ndim=2] pos,
    float lx, float ly, float lz, float d,
    np.ndarray[np.int32_t, ndim=1] orig_idx,
    np.ndarray[np.int32_t, ndim=1] ext_idx
):
    cdef int N = pos.shape[0]
    cdef int max_extended = 2 * N
    cdef np.ndarray[np.float32_t, ndim=2] ext_pos = np.empty((max_extended, 3), dtype=np.float32)
    ext_pos[:N, :] = pos
    ext_idx[:N] = orig_idx
    cdef int count = N

    cdef np.ndarray mask_x = pos[:, 0] < d
    cdef np.ndarray mask_y = pos[:, 1] < d
    cdef np.ndarray mask_z = pos[:, 2] < d
    cdef np.ndarray[np.float32_t, ndim=1] shift_vec
    cdef np.ndarray[np.float32_t, ndim=2] new_pos
    cdef np.ndarray[np.int32_t, ndim=1] new_idx
    cdef int n_new

    shifts = [
        (mask_x, [lx, 0.0, 0.0]),
        (mask_y, [0.0, ly, 0.0]),
        (mask_z, [0.0, 0.0, lz]),
        (mask_x & mask_y, [lx, ly, 0.0]),
        (mask_x & mask_z, [lx, 0.0, lz]),
        (mask_y & mask_z, [0.0, ly, lz]),
        (mask_x & mask_y & mask_z, [lx, ly, lz]),
    ]
    for mask, shift in shifts:
        shift_vec = np.array(shift, dtype=np.float32)
        new_idx = orig_idx[mask]
        if new_idx.shape[0] > 0:
            new_pos = pos[mask] + shift_vec
            n_new = new_pos.shape[0]
            if count + n_new > max_extended:
                raise RuntimeError("Too many extended atoms.")
            ext_pos[count:count + n_new, :] = new_pos
            ext_idx[count:count + n_new] = new_idx
            count += n_new
    return ext_pos[:count]

@cython.nogil
cpdef np.ndarray[np.float32_t, ndim=2] extend_triclinic(
    np.ndarray[np.float32_t, ndim=2] pos,
    float lx, float ly, float lz, float alpha, float beta, float gamma, float d,
    np.ndarray[np.int32_t, ndim=1] orig_idx,
    np.ndarray[np.int32_t, ndim=1] ext_idx
):
    cdef int N = pos.shape[0]
    cdef int max_extended = 2 * N
    cdef np.ndarray[np.float32_t, ndim=2] ext_pos = np.empty((max_extended, 3), dtype=np.float32)
    ext_pos[:N, :] = pos
    ext_idx[:N] = orig_idx
    cdef int count = N

    cdef float PI = 3.14159265
    cdef float a_rad = alpha * PI / 180.0
    cdef float b_rad = beta * PI / 180.0
    cdef float g_rad = gamma * PI / 180.0
    cdef float c_x, c_y, c_z, c_sq, V
    cdef np.ndarray[np.float32_t, ndim=2] A, A_inv
    cdef np.ndarray[np.float32_t, ndim=2] f
    cdef np.ndarray[np.float32_t, ndim=1] T
    cdef int i, i0, i1, i2, n0, n1, n2
    cdef int shifts0[3], shifts1[3], shifts2[3]
    cdef float translation_x, translation_y, translation_z

    a_vec = np.array([lx, 0.0, 0.0], dtype=np.float32)
    b_vec = np.array([ly * np.cos(g_rad), ly * np.sin(g_rad), 0.0], dtype=np.float32)
    c_x = lz * np.cos(b_rad)
    c_y = lz * ((np.cos(a_rad) - np.cos(b_rad) * np.cos(g_rad)) / np.sin(g_rad))
    c_sq = lz * lz - c_x * c_x - c_y * c_y
    c_z = sqrt(c_sq) if c_sq > 0 else 0.0
    c_vec = np.array([c_x, c_y, c_z], dtype=np.float32)

    A = np.vstack((a_vec, b_vec, c_vec)).astype(np.float32)
    A_inv = np.linalg.inv(A).astype(np.float32)
    f = pos.dot(A_inv.T)
    V = abs(np.linalg.det(A))
    T = np.empty(3, dtype=np.float32)
    T[0] = V / np.linalg.norm(np.cross(A[1], A[2]))
    T[1] = V / np.linalg.norm(np.cross(A[0], A[2]))
    T[2] = V / np.linalg.norm(np.cross(A[0], A[1]))

    for i in range(N):
        shifts0[0] = 0; n0 = 1
        shifts1[0] = 0; n1 = 1
        shifts2[0] = 0; n2 = 1
        if f[i, 0] * T[0] < d:
            shifts0[n0] = 1; n0 += 1
        if (1.0 - f[i, 0]) * T[0] < d:
            shifts0[n0] = -1; n0 += 1
        if f[i, 1] * T[1] < d:
            shifts1[n1] = 1; n1 += 1
        if (1.0 - f[i, 1]) * T[1] < d:
            shifts1[n1] = -1; n1 += 1
        if f[i, 2] * T[2] < d:
            shifts2[n2] = 1; n2 += 1
        if (1.0 - f[i, 2]) * T[2] < d:
            shifts2[n2] = -1; n2 += 1
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    if shifts0[i0] == 0 and shifts1[i1] == 0 and shifts2[i2] == 0:
                        continue
                    translation_x = shifts0[i0]*A[0,0] + shifts1[i1]*A[1,0] + shifts2[i2]*A[2,0]
                    translation_y = shifts0[i0]*A[0,1] + shifts1[i1]*A[1,1] + shifts2[i2]*A[2,1]
                    translation_z = shifts0[i0]*A[0,2] + shifts1[i1]*A[1,2] + shifts2[i2]*A[2,2]
                    if count >= max_extended:
                        raise RuntimeError("Too many triclinic extensions.")
                    ext_pos[count, 0] = pos[i, 0] + translation_x
                    ext_pos[count, 1] = pos[i, 1] + translation_y
                    ext_pos[count, 2] = pos[i, 2] + translation_z
                    ext_idx[count] = i
                    count += 1
    return ext_pos[:count]

@cython.nogil
cpdef np.ndarray[np.uint32_t, ndim=2] build_contact_map_pbc(np.ndarray pos, box_params, float cutoff, list res_indices):
    cdef int N = pos.shape[0]
    if pos.dtype != np.float32:
        pos = pos.astype(np.float32)
    box_params = np.asarray(box_params, dtype=np.float32)
    if box_params.shape[0] != 6:
        raise ValueError("box_params must have 6 elements: [lx, ly, lz, alpha, beta, gamma]")

    cdef float lx = box_params[0]
    cdef float ly = box_params[1]
    cdef float lz = box_params[2]
    cdef float alpha = box_params[3]
    cdef float beta  = box_params[4]
    cdef float gamma = box_params[5]

    cdef bint is_ortho = (abs(alpha - 90.0) < 1e-4 and abs(beta - 90.0) < 1e-4 and abs(gamma - 90.0) < 1e-4)
    cdef np.ndarray[np.int32_t, ndim=1] orig_idx = np.arange(N, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] ext_idx = np.empty(2 * N, dtype=np.int32)

    cdef np.ndarray[np.float32_t, ndim=2] ext_pos
    if is_ortho:
        ext_pos = extend_orthorhombic(pos, lx, ly, lz, cutoff, orig_idx, ext_idx)
    else:
        ext_pos = extend_triclinic(pos, lx, ly, lz, alpha, beta, gamma, cutoff, orig_idx, ext_idx)

    cdef float[:, :] ext_view = ext_pos
    cdef np.ndarray stds = np.std(ext_pos, axis=0)
    cdef int max_dim = np.argmax(stds)

    # extract the 1D projection, make it contiguous and copyable
    cdef np.ndarray[np.float32_t, ndim=1] coord_1d = np.ascontiguousarray(ext_view[:, max_dim])

    # sort indices and project sorted coordinate view
    cdef np.ndarray[np.int32_t, ndim=1] sorted_idx = np.argsort(coord_1d).astype(np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] sorted_coord_1d = coord_1d[sorted_idx]
    cdef float[:] coord_1d_view = sorted_coord_1d

    # cdef float density_1d = len(ext_pos) / (coord_1d_view[len(coord_1d_view) - 1] - coord_1d_view[0])
    # cdef int max_pairs = <int>((cutoff * density_1d) ** 2 / 10)
    cdef int max_pairs = (N * N)
    cdef np.ndarray[np.int32_t, ndim=2] pairs_ij = filter_pairs_1d(ext_view, coord_1d_view, sorted_idx, cutoff, max_pairs)
    cdef np.ndarray i_idx = ext_idx[pairs_ij[:, 0]]
    cdef np.ndarray j_idx = ext_idx[pairs_ij[:, 1]]
    cdef np.ndarray row = np.concatenate([i_idx, j_idx, np.arange(N)])
    cdef np.ndarray col = np.concatenate([j_idx, i_idx, np.arange(N)])
    cdef np.ndarray data = np.ones(row.shape[0], dtype=np.int8)

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