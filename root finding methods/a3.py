import numpy as np

def evaluate_polynomial(coefficients, x):
    return np.polyval(coefficients, x)

def bisection_method(coefficients, a=None, b=None, tol=1e-7, max_iter=100000):
    if a is None or b is None:
        raise ValueError("Both interval endpoints 'a' and 'b' must be provided.")
    
    a = np.float32(a)
    b = np.float32(b)
    tol = np.float32(tol)
    
    fa = evaluate_polynomial(coefficients, a)
    fb = evaluate_polynomial(coefficients, b)
    
    if np.sign(fa) == np.sign(fb):
        raise ValueError("The function must have opposite signs at a and b.")

    iteration = 0
    while iteration < max_iter:
        c = np.float32((a + b) / 2.0)  # Midpoint
        fc = evaluate_polynomial(coefficients, c)
        
        if np.abs(fc) < tol or np.abs(b - a) < tol:
            return c
        
        # Decide the new interval
        if np.sign(fc) == np.sign(fa):
            a = c  # Root is in [c, b]
            fa = fc
        else:
            b = c  # Root is in [a, c]
            fb = fc
        
        iteration += 1
    
    raise RuntimeError("Max iterations exceeded without finding root.")
