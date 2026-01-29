import json
import requests
from tqdm import tqdm

# Load Test Data
with open("./data/eval_dataset.json", "r") as f:
    test_cases = json.load(f)

print(f"Running evaluation on {len(test_cases)} cases...")

results = []
correct_count = 0

for case in tqdm(test_cases):
    log_line = case["input"]
    expected_json_str = case["expected_output"]
    expected_dict = json.loads(expected_json_str)
    
    # Call Local API
    try:
        resp = requests.post("http://192.168.1.42:8005/parse", json={"log_line": log_line})
        actual_output = resp.json().get("parsed_json", "{}")
        
        # Validation Logic
        try:
            actual_dict = json.loads(actual_output)
            
            # Simple Scoring: Check if key fields match
            match = True
            if actual_dict.get("src_ip") != expected_dict.get("src_ip"): match = False
            if actual_dict.get("user") != expected_dict.get("user"): match = False
            
            if match:
                correct_count += 1
            
            results.append({
                "input": log_line,
                "expected": expected_dict,
                "actual": actual_dict,
                "match": match
            })
            
        except json.JSONDecodeError:
            results.append({"input": log_line, "error": "Invalid JSON produced", "raw": actual_output})

    except Exception as e:
        print(f"API Error: {e}")

accuracy = (correct_count / len(test_cases)) * 100
print(f"\nFinal Accuracy: {accuracy:.2f}%")

# DEBUG BLOCK
print("\n--- DEBUG: Top 3 Failures ---")
for i, res in enumerate(results[:3]):
    if not res.get("match", False):
        print(f"\nCase #{i+1}:")
        print(f"Expected: {res.get('expected')}")
        print(f"Actual (Raw): {res.get('actual')}")
        print(f"Error (if any): {res.get('error')}")
        print(f"Raw String from API: {res.get('raw')}")