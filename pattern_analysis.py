import json
import numpy as np

def detailed_analysis():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    print("Cases with very low receipts (to understand base day/mile rates):")
    print("Days  Miles   Receipts   Expected   $/Day    $/Mile")
    print("-" * 50)
    
    low_receipt_cases = []
    for case in data:
        inp = case['input']
        if inp['total_receipts_amount'] < 25:  
            low_receipt_cases.append(case)
    
    for case in low_receipt_cases[:15]:
        inp = case['input']
        expected = case['expected_output']
        
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        
        per_day = expected / days if days > 0 else 0
        per_mile = expected / miles if miles > 0 else 0
        
        print(f"{days:4d} {miles:7.1f} {receipts:10.2f} {expected:10.2f} {per_day:8.2f} {per_mile:8.2f}")

    print("\nCases with very low miles (to understand day rates):")
    print("Days  Miles   Receipts   Expected   $/Day")
    print("-" * 40)
    
    low_mile_cases = []
    for case in data:
        inp = case['input']
        if inp['miles_traveled'] < 50:  # Very low miles
            low_mile_cases.append(case)
    
    for case in low_mile_cases[:10]:
        inp = case['input']
        expected = case['expected_output']
        
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        
        per_day = expected / days if days > 0 else 0
        
        print(f"{days:4d} {miles:7.1f} {receipts:10.2f} {expected:10.2f} {per_day:8.2f}")

if __name__ == "__main__":
    detailed_analysis()
