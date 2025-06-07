import json
import subprocess
import statistics

def test_implementation(num_cases=50):
    """Test the implementation on the first num_cases"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    errors = []
    exact_matches = 0
    close_matches = 0
    
    print(f"Testing implementation on first {num_cases} cases...")
    
    for i, case in enumerate(data[:num_cases]):
        if i % 10 == 0:
            print(f"Processing case {i+1}/{num_cases}...")
            
        inp = case['input']
        expected = case['expected_output']
        
        try:
            result = subprocess.run([
                './run.sh', 
                str(inp['trip_duration_days']),
                str(inp['miles_traveled']),
                str(inp['total_receipts_amount'])
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                predicted = float(result.stdout.strip())
                error = abs(predicted - expected)
                errors.append(error)
                
                if error <= 0.01:
                    exact_matches += 1
                elif error <= 1.00:
                    close_matches += 1
                    
                if i < 10 or error > 50: 
                    print(f"  Case {i+1}: Expected ${expected:.2f}, Got ${predicted:.2f}, Error ${error:.2f}")
            else:
                print(f"  Case {i+1}: ERROR - {result.stderr}")
                errors.append(1000)  
                
        except Exception as e:
            print(f"  Case {i+1}: EXCEPTION - {e}")
            errors.append(1000)
    
    if errors:
        avg_error = statistics.mean(errors)
        median_error = statistics.median(errors)
        
        print(f"\nResults on {num_cases} cases:")
        print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/num_cases*100:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches} ({close_matches/num_cases*100:.1f}%)")
        print(f"  Average error: ${avg_error:.2f}")
        print(f"  Median error: ${median_error:.2f}")
    else:
        print("No successful predictions!")

if __name__ == "__main__":
    test_implementation()
