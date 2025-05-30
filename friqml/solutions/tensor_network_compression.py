import numpy as np
from tqdm import tqdm


# EXERCISE 1
def _transpose_and_reshape(img, nbit):
    """Transposes and reshapes the input image tensor for MPS compression."""
  # prepare the original tensor
    data = np.reshape(img, [2]*(2*nbit))
    dims = []
    for i in range(nbit):
        dims.append(i)
        dims.append(i+nbit)
    data1 = np.transpose(data, dims)
    A = np.reshape(data1, [4]*nbit)
    return A


transpose_and_reshape = _transpose_and_reshape


# EXERCISE 2
def _mps_compress(A, nbit, Dmax):
    """Compresses the tensor using MPS with a given maximum bond dimension."""
    mps = []
    dr = 1
    npar = 0
    for i in tqdm(range(nbit-1)):
        A = np.reshape(A, [4*dr, 4**(nbit-1-i)])
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        dl = dr
        dr = np.min([len(s), Dmax])
        u = np.reshape(u[:, :dr], [dl, 4, dr])
        A = np.einsum("i,ij->ij", s[:dr], vh[:dr])
        mps.append(u)
        npar += u.size
    npar += A.size
    mps.append(np.reshape(A, [dr, 4, 1]))
    return mps, npar


mps_compress = _mps_compress

# EXERCISE 3


def _inverse_transpose_and_reshape(mps, nbit):
    """Restores the image from the MPS format."""
    idims = [2*i for i in range(nbit)]+[2*i+1 for i in range(nbit)]
    data_c = mps[0]
    for i in range(1, nbit):
        data_c = np.einsum("...i,ijk->...jk", data_c, mps[i])
    data_c = np.reshape(data_c, [2]*(2*nbit))
    data_c = np.transpose(data_c, idims)
    size = 2**nbit
    img_c = np.reshape(data_c, [size, size])
    return img_c


inverse_transpose_and_reshape = _inverse_transpose_and_reshape


# EXERCISE 4
def _compress_image(img, nbit=12, Dmax=64):
    """Compresses an image using Matrix Product States (MPS)."""
    A = _transpose_and_reshape(img, nbit)
    mps, npar = _mps_compress(A, nbit=nbit, Dmax=Dmax)
    img_c = _inverse_transpose_and_reshape(mps, nbit=nbit)
    return img_c, npar, mps


compress_image = _compress_image
