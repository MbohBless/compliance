#!/bin/bash

# Black Box Challenge - ML-Based Implementation
# Uses k-nearest neighbors approach
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

python3 calculate_reimbursement.py "$1" "$2" "$3"

