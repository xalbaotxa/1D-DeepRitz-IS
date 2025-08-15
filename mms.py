import sympy as sy
from sympy.parsing.sympy_parser import parse_expr
import inspect

def convert_to_jax(func, x):
    lfunc = sy.lambdify(x,func,modules='jax',printer=sy.printing.numpy.JaxPrinter)
    source_code = inspect.getsource(lfunc)
    jax_code = source_code.split('return')[-1].strip()
    for i in range(len(x)):
        jax_code = jax_code.replace('x%d' % i,'x[%d]' % i)
    jax_code = 'lambda x : ' + jax_code
    return jax_code

def get_gdim_from_str(string):
    if   bool(string.find('x[2]')): return 3
    elif bool(string.find('x[1]')): return 2
    elif bool(string.find('x[0]')): return 1
    else: raise ValueError('Could not extract geometric dimension from string!')

def get_BC_function(gdim):
    if gdim == 1:
        out = lambda x : 4*x[0]*(1-x[0])
    elif gdim == 2:
        out = lambda x : 16*x[0]*(1-x[0])*x[1]*(1-x[1])
    else:
        out = lambda x : 64*x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])
    return out

def apply_mms(uex_str, verbose=False):

    gdim = get_gdim_from_str(uex_str)

    x = sy.symbols('x:%d' % gdim)
    uex = parse_expr(uex_str, local_dict={'x':x})

    def laplacian(u):
        return sum([sy.diff(u, x[i], 2) for i in range(gdim)])

    def get_mms_rhs(u):
        return -laplacian(u)

    rhs = get_mms_rhs(uex)
    problem_data = {'u_ex' : convert_to_jax(uex,x), 'f' : convert_to_jax(rhs,x)}

    if verbose:
        print(problem_data['u_ex'])
        print(problem_data['f'])

    return problem_data

if __name__ == '__main__':
    uex1D = 'sin(10*pi*x[0])*x[0]'
    uex2D = 'sin(10*pi*x[0])*x[0]*x[1]*sin(4*pi*x[1]) + exp(-4*((x[0]-0.5)**2 + (x[1]-0.5)**2))*16*x[0]*(1-x[0])*x[1]*(1-x[1])'

    apply_mms(uex1D, verbose=True)
    apply_mms(uex2D, verbose=True)

