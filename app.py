from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp
from typing import Callable, Tuple, List, Dict

app = Flask(__name__)

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
                'a': round(a, 6),
                'b': round(b, 6),
                'c': round(c, 6),
                'f(c)': round(fc, 6),
                'error': round(error, 8)
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
                'a': round(a, 6),
                'b': round(b, 6),
                'c': round(c, 6),
                'f(c)': round(fc, 6),
                'error': round(error, 8)
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
                'x0': round(x0, 6),
                'x1': round(x1, 6),
                'x2': round(x2, 6),
                'f(x2)': round(f2, 6),
                'error': round(error, 8)
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
                'x': round(x, 6),
                'f(x)': round(fx, 6),
                "f'(x)": round(dfx, 6),
                'x_new': round(x_new, 6),
                'error': round(error, 8)
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
                'x': round(x, 6),
                'g(x)': round(x_new, 6),
                'error': round(error, 8)
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
                'x': round(x, 6),
                'f(x)': round(fx, 6),
                'f(x+δx)': round(fx_delta, 6),
                'x_new': round(x_new, 6),
                'error': round(error, 8)
            })
            
            if error < tol or abs(f(x_new)) < tol:
                return x_new, iterations
            
            x = x_new
        
        return x, iterations


solver = ZOFSolver()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        method = data['method']
        equation = data['equation']
        tol = float(data['tolerance'])
        max_iter = int(data['max_iterations'])
        
        f = solver.parse_equation(equation)
        
        if method == 'bisection':
            a = float(data['a'])
            b = float(data['b'])
            root, iterations = solver.bisection_method(f, a, b, tol, max_iter)
            
        elif method == 'regula_falsi':
            a = float(data['a'])
            b = float(data['b'])
            root, iterations = solver.regula_falsi_method(f, a, b, tol, max_iter)
            
        elif method == 'secant':
            x0 = float(data['x0'])
            x1 = float(data['x1'])
            root, iterations = solver.secant_method(f, x0, x1, tol, max_iter)
            
        elif method == 'newton_raphson':
            x0 = float(data['x0'])
            df = solver.parse_derivative(equation)
            root, iterations = solver.newton_raphson_method(f, df, x0, tol, max_iter)
            
        elif method == 'fixed_point':
            g_eq = data['g_equation']
            g = solver.parse_equation(g_eq)
            x0 = float(data['x0'])
            root, iterations = solver.fixed_point_iteration(g, x0, tol, max_iter)
            
        elif method == 'modified_secant':
            x0 = float(data['x0'])
            delta = float(data['delta'])
            root, iterations = solver.modified_secant_method(f, x0, delta, tol, max_iter)
        
        else:
            return jsonify({'error': 'Invalid method'}), 400
        
        return jsonify({
            'success': True,
            'root': round(root, 10),
            'iterations': iterations,
            'num_iterations': len(iterations),
            'final_error': iterations[-1]['error'],
            'f_root': round(float(f(root)), 10)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)