import json
import numpy as np

def simple_regression():
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    X = []
    y = []
    
    for case in data:
        inp = case['input']
        X.append([
            inp['trip_duration_days'],
            inp['miles_traveled'],
            inp['total_receipts_amount']
        ])
        y.append(case['expected_output'])
    
    X = np.array(X)
    y = np.array(y)
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    try:
        beta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
        
        print("Simple Linear Regression Coefficients:")
        print(f"  Intercept: {beta[0]:.2f}")
        print(f"  Days coefficient: {beta[1]:.2f}")
        print(f"  Miles coefficient: {beta[2]:.4f}")
        print(f"  Receipts coefficient: {beta[3]:.4f}")
        
        print("\nTesting on first 10 cases:")
        print("Days Miles  Receipts  Expected  Predicted   Error")
        print("-" * 50)
        
        total_error = 0
        for i in range(min(10, len(X))):
            predicted = beta[0] + beta[1]*X[i,0] + beta[2]*X[i,1] + beta[3]*X[i,2]
            error = abs(predicted - y[i])
            total_error += error
            
            print(f"{X[i,0]:4.0f} {X[i,1]:5.0f} {X[i,2]:9.2f} {y[i]:9.2f} {predicted:9.2f} {error:8.2f}")
        
        avg_error = total_error / 10
        print(f"\nAverage error on first 10 cases: ${avg_error:.2f}")
        
        return beta
        
    except np.linalg.LinAlgError:
        print("Matrix is singular, cannot solve")
        return None

if __name__ == "__main__":
    coeffs = simple_regression()
