import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from scipy.special import roots_legendre
from scipy.stats import qmc

from jax import config
config.update("jax_enable_x64", True)

class QuadratureMethod(object):
    def __init__(self, gdim, seed=42):
        self.gdim = gdim
        self.key = jax.random.PRNGKey(seed)
        self.RNG = np.random.default_rng(seed)
        self.qmc_sampler = qmc.Sobol(d=gdim, scramble=True, seed=self.RNG)

    def MC(self, n):
        xtrain = self.RNG.uniform(size=(self.gdim, n))
        weights = jnp.ones((n,))/n
        return jnp.array(xtrain), weights

    def QMC(self, n):
        m = int(np.ceil(np.log2(n)))
        xtrain = self.qmc_sampler.random_base2(m=m).reshape((self.gdim,-1))
        self.qmc_sampler.reset()
        self.qmc_sampler._scramble()
        weights = jnp.ones((2**m,))*(2.**-m)
        return jnp.array(xtrain), weights

    def FixedGaussQuad(self, n):
        gdim = self.gdim
        x,w = roots_legendre(n)
        x = (x+1)/2
        w /= 2
        if gdim > 1:
            X = np.meshgrid(*[x]*gdim)
            x = np.vstack([item.flatten() for item in X])
            w = np.outer(w,w).flatten() if gdim == 2 else np.einsum('i,j,k->ijk', *[w]*3).flatten()
        else:
            x = x.reshape((1,-1))
        return jnp.array(x),jnp.array(w)

    def ISConstant(self, n, M, f_values):
        xtrain = jnp.array(self.RNG.uniform(size=(self.gdim, n)))
        f_sum = jnp.sum(f_values)
        cdf = jnp.cumsum(f_values) / f_sum
        mu = 1 / M * f_sum
        xtrain = jax.vmap(lambda x: self.F_inv(x, M, f_values, mu, cdf))(xtrain)       
        i_indices = (xtrain * M).astype(int)  # Compute i for all samples
        weights = mu / (n * f_values[i_indices[0,:]])
        effective_size = jnp.sum(weights)**2 / jnp.sum(weights**2)
        return xtrain, weights, effective_size
        

    @eqx.filter_jit
    def ISConstant2(self, n, x_values, f_values, xtrain):
        x_values = jnp.insert(x_values, 0, 0, axis=1)
        sizes = x_values[0,1:]-x_values[0,:-1]
        mu = jnp.sum(f_values * sizes)
        cdf = jnp.cumsum(f_values * sizes) / mu
        i_indices, xtrain = jax.vmap(lambda x: self.F_inv(x, x_values, f_values, mu, cdf))(xtrain)     
        weights = mu / (n * f_values[i_indices[0,:]])
        effective_size = jnp.sum(weights)**2 / jnp.sum(weights**2)
        return xtrain, weights, effective_size
    
    @eqx.filter_jit
    def ISConstant3(self, key, n, a, edges, f_values):
        key_main, key_mix, key_uniform, key_piece = jax.random.split(key, 4)
        from_uniform = jax.random.bernoulli(key_mix, 1-a, shape=(n,))
        uniform_samples = jax.random.uniform(key_uniform, shape=(n,))

        M = f_values.shape[0]
        widths = edges[1:] - edges[:-1]
        mu = jnp.sum(f_values * widths)
        probs = f_values * widths / mu
        interval_indices = jax.random.choice(key_piece, a=M, shape=(n,), p=probs)
        lefts = edges[interval_indices]
        rights = edges[interval_indices + 1]
        u = jax.random.uniform(key_main, shape=(n,))
        piecewise_samples = lefts + (rights - lefts) * u
        samples = jnp.where(from_uniform, uniform_samples, piecewise_samples)
        def eval_q(x):
            idx = jnp.searchsorted(edges, x, side='right') - 1
            q_piece = f_values[idx] / mu
            return (1-a) * 1.0 + a * q_piece, idx

        q_vals, idx = jax.vmap(eval_q)(samples)
        return samples[None, :], 1 / (q_vals * n), idx
        

    @eqx.filter_jit
    def F_inv(self, y, x_values, f_values, mu, cdf):
        i = jnp.searchsorted(cdf, y)
        return i, jnp.where(y <= cdf[0], mu * y / f_values[0], x_values[0,i] + mu * (y - cdf[i-1]) / f_values[i]) 

    @eqx.filter_jit
    def ISM(self, n, a, x_values, f_values, xtrain):
        x_values = jnp.concatenate([jnp.array([-x_values[0,0]]), x_values[0,:], jnp.array([2 - x_values[0,-1]])])
        gridpoints = (x_values[:-1] + x_values[1:]) / 2
        sizes = jnp.diff(gridpoints)
        mu = jnp.sum(f_values * sizes)
        cdf = jnp.cumsum(f_values * sizes) / mu
        i_indices, xtrain = jax.vmap(lambda x: self.F_inv2(x, a, gridpoints, f_values, mu, cdf))(xtrain)        
        w = 1 / (a * f_values[i_indices[0,:]] / mu +1 -a) / n
        return xtrain, w, i_indices

    @eqx.filter_jit
    def F_inv2(self, y, a, gridpoints, f_values, mu, cdf):
        thresholds = a * cdf + (1 - a) * gridpoints[1:]
        i = jnp.searchsorted(thresholds, y)
        padded_cdf = jnp.concatenate([jnp.array([0.0]), cdf])  
        padded_gridpoints = jnp.concatenate([jnp.array([0.0]), gridpoints])  
        f_i = f_values[i] 
        numerator = y - a * (padded_cdf[i] - f_i * padded_gridpoints[i+1] / mu)
        denominator = 1 - a + a * f_i / mu
        return i, numerator / denominator

    @eqx.filter_jit
    def ISMemory(self, n, a, alpha, x_values, f_values, xtrain):
        memory = f_values.shape[0]
        left  = -x_values[:, 0]      
        right =  2 - x_values[:, -1]  
        x_padded = jnp.concatenate([left[:, None], x_values, right[:, None]], axis=1)  
        gridpoints = (x_padded[:, :-1] + x_padded[:, 1:]) / 2
        sizes = jnp.diff(gridpoints, axis=1)
        mu = jnp.sum(f_values * sizes, axis=1)
        cdf = jnp.cumsum(f_values * sizes, axis=1) / mu[:, jnp.newaxis]
        p = alpha * (1 - alpha) ** (memory - jnp.arange(1, memory+1))
        p = p / jnp.sum(p)
        thresholds = jnp.cumsum(p)
        xtrain_flat = xtrain[0, :]
        q_index = jnp.sum(xtrain_flat[:, None] >= thresholds[None, :], axis=1)
        q_index = jnp.squeeze(q_index)
        def apply_qn(y, i):
            return self.F_inv2(y, a, gridpoints[i], f_values[i], mu[i], cdf[i])

        i, transformed_xtrain = jax.vmap(apply_qn)(xtrain[0,:], q_index)
        w = 1 / (a * f_values[q_index, i] / mu[q_index] +1 -a) / n
        return transformed_xtrain.reshape(1, -1), w

    @eqx.filter_jit
    def F_inv3(self, y, x_values, f_values, mu, cdf):
        i = jnp.searchsorted(cdf, y)
        return i, jnp.where(y <= cdf[0], mu * y / f_values[0], x_values[i] + mu * (y - cdf[i-1]) / f_values[i]) 
