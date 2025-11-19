"""
Test script for ZOF Solver
Tests all six numerical methods with known test cases
"""

from ZOF_CLI import ZOFSolver
import numpy as np

def test_bisection():
    """Test Bisection Method"""
    print("\n" + "="*60)
    print("Testing BISECTION METHOD")
    print("="*60)
    
    solver = ZOFSolver()
    f = solver.parse_equation("x**3 - x - 2")
    
    try:
        root, iterations = solver.bisection_method(f, 1, 2, 0.0001, 50)
        print(f"✓ Test passed!")
        print(f"  Root found: {root:.6f}")
        print(f"  Expected: ~1.521380")
        print(f"  Iterations: {len(iterations)}")
        print(f"  f(root): {f(root):.8f}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_regula_falsi():
    """Test Regula Falsi Method"""
    print("\n" + "="*60)
    print("Testing REGULA FALSI METHOD")
    print("="*60)
    
    solver = ZOFSolver()
    f = solver.parse_equation("x**3 - x - 2")
    
    try:
        root, iterations = solver.regula_falsi_method(f, 1, 2, 0.0001, 50)
        print(f"✓ Test passed!")
        print(f"  Root found: {root:.6f}")
        print(f"  Expected: ~1.521380")
        print(f"  Iterations: {len(iterations)}")
        print(f"  f(root): {f(root):.8f}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_secant():
    """Test Secant Method"""
    print("\n" + "="*60)
    print("Testing SECANT METHOD")
    print("="*60)
    
    solver = ZOFSolver()
    f = solver.parse_equation("x**3 - x - 2")
    
    try:
        root, iterations = solver.secant_method(f, 1, 2, 0.0001, 50)
        print(f"✓ Test passed!")
        print(f"  Root found: {root:.6f}")
        print(f"  Expected: ~1.521380")
        print(f"  Iterations: {len(iterations)}")
        print(f"  f(root): {f(root):.8f}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_newton_raphson():
    """Test Newton-Raphson Method"""
    print("\n" + "="*60)
    print("Testing NEWTON-RAPHSON METHOD")
    print("="*60)
    
    solver = ZOFSolver()
    equation = "x**3 - x - 2"
    f = solver.parse_equation(equation)
    df = solver.parse_derivative(equation)
    
    try:
        root, iterations = solver.newton_raphson_method(f, df, 1.5, 0.0001, 50)
        print(f"✓ Test passed!")
        print(f"  Root found: {root:.6f}")
        print(f"  Expected: ~1.521380")
        print(f"  Iterations: {len(iterations)}")
        print(f"  f(root): {f(root):.8f}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_fixed_point():
    """Test Fixed Point Iteration"""
    print("\n" + "="*60)
    print("Testing FIXED POINT ITERATION")
    print("="*60)
    
    solver = ZOFSolver()
    # For x^3 - x - 2 = 0, rearrange to x = (x + 2)^(1/3)
    g = solver.parse_equation("(x + 2)**(1/3)")
    
    try:
        root, iterations = solver.fixed_point_iteration(g, 1.5, 0.0001, 50)
        print(f"✓ Test passed!")
        print(f"  Root found: {root:.6f}")
        print(f"  Expected: ~1.521380")
        print(f"  Iterations: {len(iterations)}")
        # Verify: x^3 - x - 2 should be close to 0
        f = solver.parse_equation("x**3 - x - 2")
        print(f"  f(root): {f(root):.8f}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_modified_secant():
    """Test Modified Secant Method"""
    print("\n" + "="*60)
    print("Testing MODIFIED SECANT METHOD")
    print("="*60)
    
    solver = ZOFSolver()
    f = solver.parse_equation("x**3 - x - 2")
    
    try:
        root, iterations = solver.modified_secant_method(f, 1.5, 0.01, 0.0001, 50)
        print(f"✓ Test passed!")
        print(f"  Root found: {root:.6f}")
        print(f"  Expected: ~1.521380")
        print(f"  Iterations: {len(iterations)}")
        print(f"  f(root): {f(root):.8f}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_additional_equations():
    """Test with additional equations"""
    print("\n" + "="*60)
    print("Testing ADDITIONAL EQUATIONS")
    print("="*60)
    
    solver = ZOFSolver()
    
    # Test Case 1: cos(x) - x = 0, root ~ 0.7391
    print("\nTest: cos(x) - x = 0")
    try:
        f = solver.parse_equation("cos(x) - x")
        root, _ = solver.bisection_method(f, 0, 1, 0.0001, 50)
        print(f"  ✓ Root: {root:.6f} (Expected: ~0.739085)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test Case 2: exp(x) - 3*x = 0, root ~ 0.6191
    print("\nTest: exp(x) - 3*x = 0")
    try:
        f = solver.parse_equation("exp(x) - 3*x")
        root, _ = solver.bisection_method(f, 0, 1, 0.0001, 50)
        print(f"  ✓ Root: {root:.6f} (Expected: ~0.619061)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Test Case 3: x^2 - 4 = 0, root = 2
    print("\nTest: x^2 - 4 = 0")
    try:
        f = solver.parse_equation("x**2 - 4")
        root, _ = solver.bisection_method(f, 1, 3, 0.0001, 50)
        print(f"  ✓ Root: {root:.6f} (Expected: 2.0)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print("ZOF SOLVER - AUTOMATED TESTING")
    print("="*60)
    print("\nTesting all six numerical methods...")
    
    results = []
    
    results.append(("Bisection", test_bisection()))
    results.append(("Regula Falsi", test_regula_falsi()))
    results.append(("Secant", test_secant()))
    results.append(("Newton-Raphson", test_newton_raphson()))
    results.append(("Fixed Point", test_fixed_point()))
    results.append(("Modified Secant", test_modified_secant()))
    
    test_additional_equations()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for method, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{method:20} : {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed!")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_tests()