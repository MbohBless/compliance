import json
from calculate_reimbursement import ReimbursementCalculator

def final_test():
    # Load test cases
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    calc = ReimbursementCalculator()
    errors = []
    exact_matches = 0
    close_matches = 0
    
    print("Final validation on all 1000 public cases...")
    
    for i, case in enumerate(data):
        if i % 100 == 0:
            print(f"Processing case {i+1}/1000...")
            
        inp = case['input']
        expected = case['expected_output']
        
        predicted = calc.calculate(
            inp['trip_duration_days'],
            inp['miles_traveled'],
            inp['total_receipts_amount']
        )
        
        error = abs(predicted - expected)
        errors.append(error)
        
        if error <= 0.01:
            exact_matches += 1
        elif error <= 1.00:
            close_matches += 1
    
    avg_error = sum(errors) / len(errors)
    errors_sorted = sorted(errors)
    median_error = errors_sorted[len(errors)//2]
    max_error = max(errors)
    
    print(f"\nFinal Results on 1000 public cases:")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/1000*100:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_matches/1000*100:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Median error: ${median_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    
    print(f"\nModel performance summary:")
    print(f"  Perfect accuracy on training data: {exact_matches == 1000}")
    print(f"  Ready for submission: Yes" if exact_matches >= 950 else "No, needs improvement")

if __name__ == "__main__":
    final_test()
