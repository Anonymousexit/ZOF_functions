import numpy as np
import sympy as sp
from typing import Callable, Tuple, List, Dict

class ZOFSolver:
    """Zero of Functions Solver - Implements 6 numerical methods"""
    
    def __init__(self):
        self.x = sp.Symbol('x')
        
    def parse_equation(self, equation_str: str) -> Callable:
        """Convert string equation to callable function"""
        expr = sp.sympify(equation_str)
        return sp.lambdify(self.x, expr, 'numpy')
    
    def parse_derivative(self, equation_str: str) -> Callable:
        """Compute and return derivative as callable"""
        expr = sp.sympify(equation_str)
        derivative = sp.diff(expr, self.x)
        return sp.lambdify(self.x, derivative, 'numpy')
    
    def bisection_method(self, f: Callable, a: float, b: float, 
                        tol: float, max_iter: int) -> Tuple[float, List[Dict]]:
        """Bisection Method"""
        iterations = []
        
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)
            fa = f(a)
            
            error = abs(b - a) / 2
            
            iterations.append({
                'iteration': i + 1,
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'error': error
            })
            
            if error < tol or abs(fc) < tol:
                return c, iterations
            
            if fa * fc < 0:
                b = c
            else:
                a = c
        
        return c, iterations
    
    def regula_falsi_method(self, f: Callable, a: float, b: float,
                           tol: float, max_iter: int) -> Tuple[float, List[Dict]]:
        """Regula Falsi (False Position) Method"""
        iterations = []
        
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        c_old = a
        
        for i in range(max_iter):
            fa = f(a)
            fb = f(b)
            
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)
            
            error = abs(c - c_old) if i > 0 else abs(b - a)
            
            iterations.append({
                'iteration': i + 1,
                'a': a,
                'b': b,
                'c': c,
                'f(c)': fc,
                'error': error
            })
            
            if error < tol or abs(fc) < tol:
                return c, iterations
            
            if fa * fc < 0:
                b = c
            else:
                a = c
            
            c_old = c
        
        return c, iterations
    
    def secant_method(self, f: Callable, x0: float, x1: float,
                     tol: float, max_iter: int) -> Tuple[float, List[Dict]]:
        """Secant Method"""
        iterations = []
        
        for i in range(max_iter):
            f0 = f(x0)
            f1 = f(x1)
            
            if abs(f1 - f0) < 1e-12:
                raise ValueError("Division by zero in secant method")
            
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = f(x2)
            
            error = abs(x2 - x1)
            
            iterations.append({
                'iteration': i + 1,
                'x0': x0,
                'x1': x1,
                'x2': x2,
                'f(x2)': f2,
                'error': error
            })
            
            if error < tol or abs(f2) < tol:
                return x2, iterations
            
            x0, x1 = x1, x2
        
        return x2, iterations
    
    def newton_raphson_method(self, f: Callable, df: Callable, x0: float,
                             tol: float, max_iter: int) -> Tuple[float, List[Dict]]:
        """Newton-Raphson Method"""
        iterations = []
        
        x = x0
        
        for i in range(max_iter):
            fx = f(x)
            dfx = df(x)
            
            if abs(dfx) < 1e-12:
                raise ValueError("Derivative too close to zero")
            
            x_new = x - fx / dfx
            error = abs(x_new - x)
            
            iterations.append({
                'iteration': i + 1,
                'x': x,
                'f(x)': fx,
                'f\'(x)': dfx,
                'x_new': x_new,
                'error': error
            })
            
            if error < tol or abs(f(x_new)) < tol:
                return x_new, iterations
            
            x = x_new
        
        return x, iterations
    
    def fixed_point_iteration(self, g: Callable, x0: float,
                             tol: float, max_iter: int) -> Tuple[float, List[Dict]]:
        """Fixed Point Iteration Method"""
        iterations = []
        
        x = x0
        
        for i in range(max_iter):
            x_new = g(x)
            error = abs(x_new - x)
            
            iterations.append({
                'iteration': i + 1,
                'x': x,
                'g(x)': x_new,
                'error': error
            })
            
            if error < tol:
                return x_new, iterations
            
            x = x_new
        
        return x, iterations
    
    def modified_secant_method(self, f: Callable, x0: float, delta: float,
                              tol: float, max_iter: int) -> Tuple[float, List[Dict]]:
        """Modified Secant Method"""
        iterations = []
        
        x = x0
        
        for i in range(max_iter):
            fx = f(x)
            fx_delta = f(x + delta * x)
            
            denominator = fx_delta - fx
            
            if abs(denominator) < 1e-12:
                raise ValueError("Division by zero in modified secant method")
            
            x_new = x - (delta * x * fx) / denominator
            error = abs(x_new - x)
            
            iterations.append({
                'iteration': i + 1,
                'x': x,
                'f(x)': fx,
                'f(x+δx)': fx_delta,
                'x_new': x_new,
                'error': error
            })
            
            if error < tol or abs(f(x_new)) < tol:
                return x_new, iterations
            
            x = x_new
        
        return x, iterations


def print_iterations(iterations: List[Dict], method_name: str):
    """Print iteration details in a formatted table"""
    print(f"\n{'='*80}")
    print(f"{method_name} - Iteration Details")
    print(f"{'='*80}")
    
    if not iterations:
        return
    
    # Print header
    keys = list(iterations[0].keys())
    header = " | ".join(f"{k:^12}" for k in keys)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for iteration in iterations:
        row = " | ".join(f"{str(v):^12.6f}" if isinstance(v, float) else f"{str(v):^12}" 
                        for v in iteration.values())
        print(row)


def main():
    solver = ZOFSolver()
    
    print("=" * 80)
    print("ZERO OF FUNCTIONS (ZOF) SOLVER")
    print("=" * 80)
    
    methods = {
        '1': 'Bisection Method',
        '2': 'Regula Falsi Method',
        '3': 'Secant Method',
        '4': 'Newton-Raphson Method',
        '5': 'Fixed Point Iteration',
        '6': 'Modified Secant Method'
    }
    
    print("\nAvailable Methods:")
    for key, value in methods.items():
        print(f"{key}. {value}")
    
    choice = input("\nSelect method (1-6): ")
    
    if choice not in methods:
        print("Invalid choice!")
        return
    
    print(f"\nYou selected: {methods[choice]}")
    
    try:
        equation = input("\nEnter equation f(x) (e.g., x**3 - x - 2): ")
        f = solver.parse_equation(equation)
        
        tol = float(input("Enter tolerance (e.g., 0.0001): "))
        max_iter = int(input("Enter maximum iterations (e.g., 50): "))
        
        if choice in ['1', '2']:  # Bisection or Regula Falsi
            a = float(input("Enter initial guess a: "))
            b = float(input("Enter initial guess b: "))
            
            if choice == '1':
                root, iterations = solver.bisection_method(f, a, b, tol, max_iter)
            else:
                root, iterations = solver.regula_falsi_method(f, a, b, tol, max_iter)
        
        elif choice == '3':  # Secant
            x0 = float(input("Enter initial guess x0: "))
            x1 = float(input("Enter initial guess x1: "))
            root, iterations = solver.secant_method(f, x0, x1, tol, max_iter)
        
        elif choice == '4':  # Newton-Raphson
            x0 = float(input("Enter initial guess x0: "))
            df = solver.parse_derivative(equation)
            root, iterations = solver.newton_raphson_method(f, df, x0, tol, max_iter)
        
        elif choice == '5':  # Fixed Point
            g_eq = input("Enter g(x) for fixed point iteration: ")
            g = solver.parse_equation(g_eq)
            x0 = float(input("Enter initial guess x0: "))
            root, iterations = solver.fixed_point_iteration(g, x0, tol, max_iter)
        
        elif choice == '6':  # Modified Secant
            x0 = float(input("Enter initial guess x0: "))
            delta = float(input("Enter delta (δ, e.g., 0.01): "))
            root, iterations = solver.modified_secant_method(f, x0, delta, tol, max_iter)
        
        print_iterations(iterations, methods[choice])
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Estimated Root: {root:.10f}")
        print(f"Final Error: {iterations[-1]['error']:.10e}")
        print(f"Number of Iterations: {len(iterations)}")
        print(f"f(root) = {f(root):.10e}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()