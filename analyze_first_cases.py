import json
import subprocess

def analyze_cases():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("Analyzing first 20 cases:")
    print("Days  Miles   Receipts   Expected   Predicted   Error")
    print("-" * 55)
    
    for i, case in enumerate(data[:20]):
        inp = case['input']
        expected = case['expected_output']
        
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']

        # Get prediction
        result = subprocess.run([
            'python3', 'calculate_reimbursement.py',
            str(days), str(miles), str(receipts)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            predicted = float(result.stdout.strip())
            error = abs(predicted - expected)
            
            print(f"{days:4d} {miles:7.1f} {receipts:10.2f} {expected:10.2f} {predicted:10.2f} {error:8.2f}")
        else:
            print(f"{days:4d} {miles:7.1f} {receipts:10.2f} {expected:10.2f}      ERROR")

if __name__ == "__main__":
    analyze_cases()
