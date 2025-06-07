import json
from calculate_reimbursement import ReimbursementCalculator

def simple_test():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    calc = ReimbursementCalculator()
    errors = []
    exact_matches = 0
    close_matches = 0
    
    print("Testing first 100 cases...")
    
    for i, case in enumerate(data[:100]):
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
        
        if i < 10: 
            print(f"Case {i+1:2d}: Expected ${expected:.2f}, Got ${predicted:.2f}, Error ${error:.2f}")
    
    avg_error = sum(errors) / len(errors)
    errors_sorted = sorted(errors)
    median_error = errors_sorted[len(errors)//2]
    
    print(f"\nResults on 100 cases:")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/100*100:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_matches/100*100:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Median error: ${median_error:.2f}")

if __name__ == "__main__":
    simple_test()
