import json
import numpy as np

def analyze_receipt_patterns():
  
    with open('public_cases.json', 'r') as f:
        data = json.load(f)

    # Group cases by receipt ranges to understand the patterns
    receipt_buckets = {
        'very_low': [],    # < 50
        'low': [],         # 50-200
        'medium': [],      # 200-800
        'high': [],        # 800-1500
        'very_high': []    # > 1500
    }
    
    for case in data:
        inp = case['input']
        receipts = inp['total_receipts_amount']
        
        if receipts < 50:
            receipt_buckets['very_low'].append(case)
        elif receipts < 200:
            receipt_buckets['low'].append(case)
        elif receipts < 800:
            receipt_buckets['medium'].append(case)
        elif receipts < 1500:
            receipt_buckets['high'].append(case)
        else:
            receipt_buckets['very_high'].append(case)
    
    # Analyze each bucket
    for bucket_name, cases in receipt_buckets.items():
        if not cases:
            continue
            
        print(f"\n{bucket_name.upper()} RECEIPTS ({len(cases)} cases):")
        print("Days Miles  Receipts  Expected  BaseEst  ReceiptCont  Receipt%")
        print("-" * 65)
        
        for case in cases[:10]:  
            inp = case['input']
            expected = case['expected_output']
            
            days = inp['trip_duration_days']
            miles = inp['miles_traveled']
            receipts = inp['total_receipts_amount']
            
            
            base_est = days * 120 + miles * 0.5  
            receipt_contribution = expected - base_est
            receipt_percent = (receipt_contribution / expected * 100) if expected > 0 else 0
            
            print(f"{days:4d} {miles:5.0f} {receipts:9.2f} {expected:9.2f} {base_est:8.2f} {receipt_contribution:11.2f} {receipt_percent:8.1f}%")

if __name__ == "__main__":
    analyze_receipt_patterns()
