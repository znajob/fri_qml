import numpy as np
import tensorflow as tf
from tensorflow_probability import math as tfp_math


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
        self.pi_half = tf.constant(np.pi / 2, dtype=tf.float32)

    def _binom(self, n, k):
        n = tf.cast(n, tf.float32)
        k = tf.cast(k, tf.float32)
        return tf.exp(tf.math.lgamma(n + 1) - tf.math.lgamma(k + 1) - tf.math.lgamma(n - k + 1))

    def call(self, inputs):
        x = self.flatten(inputs)  # (batch_size, N)
        x = tf.transpose(x)       # (N, batch_size)
        x = tf.cast(x, tf.float32)
        xc = tf.math.cos(x * self.pi_half)
        xs = tf.math.sin(x * self.pi_half)

        emb = []
        for j in range(self.d):
            coeff = tf.sqrt(self._binom(self.d - 1, j))
            term = coeff * tf.pow(xc, self.d - j - 1.0) * tf.pow(xs, j)
            emb.append(term)

        return tf.stack(emb, axis=-1)  # (N, batch_size, d)

# EXERCISE 3


class MPS(tf.keras.layers.Layer):
    def __init__(self, D, C, d=2, stddev=0.5):
        super(MPS, self).__init__()
        self.D = D
        self.d = d
        self.C = C
        self.stddev = stddev

    def build(self, input_shape):
        # input_shape: (N, batch_size, d)
        N = input_shape[0]
        d = input_shape[2]
        self.n = N

        self.tensor = self.add_weight(
            shape=(N, self.D, self.D, d),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev),
            trainable=True,
            name="core"
        )
        self.Aout = self.add_weight(
            shape=(self.C, self.D, self.D),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.stddev),
            trainable=True,
            name="output"
        )

    def call(self, inputs):
        # inputs: (N, batch_size, d)
        A = tf.einsum("nbi,nlri->nblr", inputs,
                      self.tensor)  # (N, batch, D, D)

        nhalf = self.n // 2
        Al = A[0, :, 0, :]  # (batch, D)
        for i in range(1, nhalf):
            Al = tf.einsum("bl,blr->br", Al, A[i])

        Ar = A[-1, :, :, 0]  # (batch, D)
        for i in range(self.n - 2, nhalf - 1, -1):
            Ar = tf.einsum("blr,br->bl", A[i], Ar)

        Aout = tf.einsum("bl,olr->bor", Al, self.Aout)  # (batch, C, D)
        out = tf.einsum("bor,br->bo", Aout, Ar)         # (batch, C)
        return out


# EXERCISE 4
class MPS_model(tf.keras.Model):
    def __init__(self, D, C=1, d=2, stddev=0.5):
        super(MPS_model, self).__init__()
        self.embedding = Embedding(d)
        self.mps = MPS(D=D, d=d, C=C, stddev=stddev)
        self.C = C

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.mps(x)
        if self.C > 1:
            return tf.keras.activations.softmax(x)
        else:
            return tf.keras.activations.sigmoid(x)