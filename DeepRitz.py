import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from time import time
from functools import partial
from jax import config
config.update("jax_enable_x64", True)

from mms import apply_mms,get_BC_function
from quadrature import QuadratureMethod

import matplotlib.pyplot as plt 

key = jax.random.PRNGKey(42)

############################ PROBLEM ####################################
gdim = 1
NMC = 50
NIS = 50
M = 100
learning_rate_MC = 1.0e-4
learning_rate_IS = 1.0e-4
a = 1.0
train_iter = 8*10**3
C = 5

exact_loss = -1/6

"""
u = lambda x: np.sin(4*np.pi * x) * x
f = lambda x: -8 * np.pi * np.cos(4 * np.pi * x) + 16 * (np.pi**2) * x * np.sin(4 * np.pi * x)
x = np.linspace(0, 1, 1000)
dx = x[1] - x[0]
u_vals = u(x)
f_vals = f(x)
du = np.gradient(u_vals, dx)
integrand = 0.5 * du**2 - u_vals * f_vals
exact_loss = np.trapezoid(integrand, x)
print('Exact loss: ', exact_loss)
"""

QM = QuadratureMethod(gdim, seed=42)
np.random.seed(42)
keys = jax.random.split(key, train_iter)
if gdim == 1:
    #uex_str = 'sin(4*pi*x[0])*x[0]'
    uex_str = '(x[0])**(1)*(1-x[0])'
    xtest = np.linspace(0, 1, 10000).reshape((1,-1))
else:
    raise NotImplementedError

problem_data = apply_mms(uex_str)
u_ex = eval(problem_data['u_ex'])
f    = eval(problem_data['f'])

bc_func = get_BC_function(gdim)

xtrain,weights = QM.QMC(NMC)

############################ MODEL AND LOSS FUNCTION ####################################

class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, key):
        #nn_dimensions = [[gdim, 1000], [1000,200], [200, 1]]
        nn_dimensions = [[gdim, 30],[30,1]]
        n_layers = len(nn_dimensions)
        keys = jax.random.split(key, n_layers)
        self.layers = [eqx.nn.Linear(nn_dimensions[i][0], nn_dimensions[i][1], key=keys[i]) for i in range(n_layers-1)]
        self.layers.append(eqx.nn.Linear(nn_dimensions[-1][0], "scalar", use_bias=False, key=keys[-1]))
        self.init_linear_weight(jax.nn.initializers.glorot_uniform, key)

    def init_linear_weight(self, init_fn, key):
        get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m) if isinstance(x, eqx.nn.Linear)]
        weights = get_weights(self)
        new_weights = [init_fn(subkey, weight.shape) for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
        self = eqx.tree_at(get_weights, self, new_weights)

    def __call__(self, x):
        phi = bc_func(x)
        for layer in self.layers[:-1]:
            x = jax.nn.sigmoid(layer(x))
        return (self.layers[-1](x)*phi).squeeze()

    def u(self, x):
        return jax.vmap(self, in_axes=1, out_axes=0)(x).squeeze()

    def du(self, x):
        return jax.vmap(jax.jacfwd(self), in_axes=1, out_axes=0)(x).T

    def hess(self, x):
        return jax.vmap(jax.jacfwd(jax.jacfwd(self.eval_basis)), in_axes=1, out_axes=0)(x).T

    def split_eval(self, *x):
        return self.eval_basis(jnp.array([*x]))

    def laplacian(self, x):
        return sum(jax.vmap(jax.jacfwd(jax.jacfwd(self.split_eval,i),i), out_axes=1)(*x) for i in range(gdim))


model = NeuralNetwork(key)

@eqx.filter_jit
def l(model, x):
    u = model.u(x)
    du = model.du(x)
    loss = jnp.sum(du**2,axis=0)/2 - f(x)*u +C
    return loss.squeeze().astype(float)

@eqx.filter_jit
def DR_lvg(model, xtrain, weights): 
    ftrain = l(model,xtrain)
    loss = ftrain@weights
    var = jnp.var(ftrain * weights)
    return loss.squeeze().astype(float)-C, (ftrain, var)

@eqx.filter_jit
def DR_lvg_with_grad(model, xtrain, weights):
    (loss, ftrain_var), grad = jax.value_and_grad(DR_lvg, has_aux=True)(model, xtrain, weights)
    ftrain, var = ftrain_var
    return loss, ftrain, var, grad

############################ TRAINING ####################################

def train(niter, use_IS, a, learning_rate, loss_value_and_grad, use_optax=True):
    key = jax.random.PRNGKey(42)
    model = NeuralNetwork(key)

    if use_optax: print("Optimizing using Optax!\n")
    else: print("Optimizing using custom solver!\n")

    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_epoch_optax(model,opt_state,xtrain,weights):
        val,ftrain,var,model_grad = loss_value_and_grad(model,xtrain,weights)
        updates, opt_state = optim.update(model_grad, opt_state, model)
        model = eqx.apply_updates(model, updates)
        flat_model_grad = jax.tree.flatten(model_grad)[0]
        grad_norm = jnp.linalg.norm(jnp.concatenate([jnp.ravel(item) for item in flat_model_grad]))
        return val,ftrain, var, grad_norm, model, opt_state

    losses = []
    grad_norms = []
    tic = time()
    var = []
    for epoch in range(niter+1):
        if use_IS == 0:
            if epoch == 0:
                print('Integrating using standard MC!')
            xtrain, weights = QM.MC(NMC)
                           
        if use_IS == 1: #unfixed mesh for proposal distribution
            if epoch == 0:
                print('Integrating using unfixed IS MC!')
                x_prev, weights = QM.MC(NMC)
                x_prev = jnp.sort(x_prev, axis=1)
                f_prev = l(model, x_prev)
                f_prev = jnp.array(f_prev)
                           
            xtrain = jnp.array(np.random.uniform(size=(1,NIS)))
            xtrain, weights, _ = QM.ISM(NIS, a, x_prev, f_prev, xtrain)

            # Sort and prepare for next iteration
            sort_indices = jnp.argsort(xtrain[0])
            xtrain = xtrain[:, sort_indices]
            weights = weights[sort_indices]
            x_prev = xtrain

        if use_IS == 2: #fixed mesh for proposal distribution
            
            if epoch == 0:
                print('Integrating using fixed IS MC!')
                x = ((jnp.arange(M)+0.5) / M).reshape(1, -1)
                edges = jnp.linspace(0, 1, M+1)
                f_values = l(model, x)
                f_values = jnp.array(f_values)
                                                   
            xtrain = jnp.array(np.random.uniform(size=(1,NIS)))
            xtrain, weights, indices = QM.ISConstant3(keys[epoch], NIS, a, edges, f_values)
            
                       
        loss, f_prev, var_value, grad_norm,model,opt_state = train_epoch_optax(model, opt_state, xtrain, weights)
        
        if use_IS == 2:
           
            # Compute sum and count per cell
            f_sums = jnp.zeros_like(f_values).at[indices].add(f_prev)
            counts = jnp.zeros_like(f_values).at[indices].add(1)

            # Avoid division by zero
            mask = counts > 0
            averaged = jnp.where(mask, f_sums / counts, f_values)

            # Update only cells that received new values
            f_values = jnp.where(mask, averaged, f_values)

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}, Variance: {var_value}')
        
        losses.append(loss)
        var.append(var_value)
        grad_norms.append(grad_norm)

    print("Training time: ", time()-tic)

    uu = model.u(xtest)
    return uu,np.array(losses),np.array(grad_norms), np.array(var)

uu,losses_optax,grad_norms, var = train(train_iter, 0, a, learning_rate_MC, loss_value_and_grad=DR_lvg_with_grad, use_optax=True)
uu_IS1,losses_optax_IS1,grad_norms_IS1, var_IS1 = train(train_iter, 1, a, learning_rate_IS, loss_value_and_grad=DR_lvg_with_grad, use_optax=True)
uu_IS2,losses_optax_IS2,grad_norms_IS2, var_IS2 = train(train_iter, 2, a, learning_rate_IS, loss_value_and_grad=DR_lvg_with_grad, use_optax=True)


############################ PLOTTING ####################################

plt.rcParams.update({
    'text.usetex': True,         # Use LaTeX for all text
    'font.family': 'serif',      # Use a serif font, which matches LaTeX default
    'font.serif': ['Computer Modern'],
    'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'legend.fontsize': 20, 'axes.titlesize': 20
})

plt.figure()

plt.plot(xtest.flatten(), uu, label=r'MC $u_{NN}$', linewidth=3)
plt.plot(xtest.flatten(), uu_IS1, label=r'IS1 $u_{NN}$', linewidth=3)
plt.plot(xtest.flatten(), uu_IS2, label=r'IS2 $u_{NN}$', linewidth=3)
plt.plot(xtest.flatten(), u_ex(xtest), linestyle='--',label=r'$u_{ex}$', linewidth=3)
plt.xlabel(r"$x$", fontsize=20)
plt.ylabel(r"$u$", fontsize=20)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(fontsize=20)

plt.figure()
x_values = range(train_iter+1)
plt.plot(x_values, var, label="MC Variance", linestyle='-', linewidth=2)
plt.plot(x_values, var_IS1, label="IS1 Variance", linestyle='-', linewidth=2)
plt.plot(x_values, var_IS2, label="IS2 Variance", linestyle='-', linewidth=2)
plt.xscale('log')
plt.xlim(10**0, train_iter)
plt.yscale('log')
plt.xlabel(r"Iteration", fontsize=20)
plt.ylabel("Variance", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.tight_layout()

plt.figure()
error = [100* jnp.sqrt(2*abs(loss_value - exact_loss)) / abs(exact_loss) for loss_value in losses_optax]
error_IS1 = [100* jnp.sqrt(2*abs(loss_value - exact_loss)) / abs(exact_loss) for loss_value in losses_optax_IS1]
error_IS2 = [100* jnp.sqrt(2*abs(loss_value - exact_loss)) / abs(exact_loss) for loss_value in losses_optax_IS2]

plt.plot(x_values, error, label="MC Relative error", linestyle='-', linewidth=2)
plt.plot(x_values, error_IS1, label="IS1 Relative error", linestyle='-', linewidth=2)
plt.plot(x_values, error_IS2, label="IS2 Relative error", linestyle='-', linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**0, train_iter)
plt.xlabel(r"Iteration", fontsize=20)
plt.ylabel(r"Relative error in $\%$", fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

