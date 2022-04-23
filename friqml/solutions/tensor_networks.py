import numpy as np
import tensorflow as tf
import scipy


# EXERCISE 1
def tensor_to_mps(A, Dmax=8):
    N = int(np.log(np.prod(A.shape))/np.log(2))
    # Write the tensor as a matrid with only 2 rows
    B = np.reshape(A, [2, -1])
    mps = []
    for i in range(N-1):
        # Apply the SVD on the matrix B representing the remaining part of the tensor which has not been yet compressed
        # The option full_matrices=False means we only get non-zero singular values
        u, s, vh = np.linalg.svd(B, full_matrices=False)
        # We take only the first Dmax singular values if there are more
        D = min([Dmax, len(s)])
        # We reshape the tensor u into the shape [Dl,2,D], where Dl is determined automatically
        # and add it to the list of MPS tensors
        mps.append(np.reshape(u[:, :D], [-1, 2, D]))
        # We calculate the new B by concatenating s and vh and then reshaping.
        B = np.einsum("i,ij->ij", s[:D], vh[:D, :])
        B = np.reshape(B, [D*2, -1])
    # Finally we only reshape the last matrix B and append it to the MPS tensors
    mps.append(np.reshape(B, [D, 2, -1]))
    return mps


def mps_to_tensor(mps):
    tensor = np.array([[1]])
    N = len(mps)
    for i in range(N):
        tensor = np.einsum("...l,lir->...ir", tensor, mps[i])
    return tensor


# EXERCISE 2
class Embedding(tf.keras.layers.Layer):
    def __init__(self, d=2):
        super(Embedding, self).__init__()
        self.d = d
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input):
        d = self.d
        x = self.flatten(input)  # (nbatch, N)
        # We want the batch size to be the second dimension
        x = tf.transpose(x)
        pi_half = np.pi/2
        xc = tf.math.cos(x*pi_half)
        xs = tf.math.sin(x*pi_half)
        emb = []
        for j in range(self.d):
            emb.append(np.sqrt(scipy.special.binom(d-1, j))
                       * xc**(d-j-1.0) * xs**(1.0*j))
        return tf.stack(emb, axis=-1)  # (N, nbatch, d)


# EXERCISE 3
class MPS(tf.keras.layers.Layer):
    def __init__(self, D, C, d=2, stddev=0.5):
        super(MPS, self).__init__()
        self.D = D
        self.d = d
        self.C = C
        self.stddev = stddev

    def build(self, input_shape):
        # We assume the input_shape is (N,nbatch,d)
        N = input_shape[0]
        d = input_shape[2]
        C = self.C
        assert d == self.d, f"Input shape should be (N,nbatch,d). Obtained feature size d={d}, expected {self.d}."

        self.n = N
        stddev = self.stddev
        D = self.D
        self.tensor = tf.Variable(tf.random.normal(
            shape=(N, D, D, d), stddev=stddev), name="tensor", trainable=True)
        self.Aout = tf.Variable(tf.random.normal(
            shape=(C, D, D), stddev=stddev), name="tensor", trainable=True)

    def call(self, input):
        # returns the log-overlap
        d = self.d
        n = len(input)
        assert d == self.d, f"Input shape should be (N,nbatch,d). Obtained feature size d={d}, expected {self.d}."
        assert n == self.n, f"Input shape should be (N,nbatch,d). Obtained input size N={n}, expected {self.n}."

        A = tf.einsum("nbi,nlri->nblr", input, self.tensor)

        nhalf = n//2
        Al = A[0, :, 0, :]
        for i in range(1, nhalf):
            Al = tf.einsum("bl,blr->br", Al, A[i])

        Ar = A[n-1, :, :, 0]
        for i in range(n-2, nhalf-1, -1):
            Al = tf.einsum("blr,br->bl", A[i], Ar)

        Aout = tf.einsum("bl,olr->bor", Al, self.Aout)
        out = tf.einsum("bor,br->bo", Aout, Ar)

        return out


# EXERCISE 4
class MPS_model(tf.keras.Model):
    def __init__(self, D, C=1, d=2, stddev=0.5):
        super(MPS_model, self).__init__()
        self.D = D
        self.d = d
        self.C = C
        self.embedding = Embedding(d)
        self.mps = MPS(D=D, d=d, C=C, stddev=0.5)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.mps(x)
        x = tf.keras.activations.sigmoid(x)
        if self.C > 1:
            x = tf.keras.activations.softmax(x)
        return x
