import json
import requests
from tqdm import tqdm
from pydantic import BaseModel, Field

# --- 1. Define Schema (The "Loom" Part) ---
class SyslogEntry(BaseModel):
    timestamp: str = Field(description="e.g. 'Jan 28 12:00:00'")
    hostname: str
    process: str
    pid: int
    event_type: str
    user: str
    src_ip: str
    port: int

# --- 2. Configuration ---
# Pointing to your NEW vLLM container for the fine-tuned model
VLLM_URL = "http://192.168.1.42:8005/v1/completions"
MODEL_NAME = "kai-log-parser"

# Updated Template with a 1-Shot Example
alpaca_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Parse the following syslog entry into structured JSON.
Example Input: Jan 28 12:00:00 firewall-1 sshd[1234]: Failed password for invalid user admin from 1.2.3.4 port 5555 ssh2
Example Response: {{"timestamp": "Jan 28 12:00:00", "hostname": "firewall-1", "process": "sshd", "pid": 1234, "event_type": "authentication_failure", "user": "admin", "src_ip": "1.2.3.4", "port": 5555}}

### Input:
{}

### Response:
"""

# --- 3. Evaluation ---
with open("data/eval_dataset.json", "r") as f:
    test_cases = json.load(f)

print(f"Running vLLM Guided Evaluation on {len(test_cases)} cases...")

correct_count = 0
results = []

for case in tqdm(test_cases):
    log_line = case["input"]
    expected_dict = json.loads(case["expected_output"])
    
    # Generate Schema from Pydantic (Loom Logic)
    json_schema = SyslogEntry.model_json_schema()

    try:
        resp = requests.post(VLLM_URL, json={
            "model": MODEL_NAME,
            "prompt": alpaca_template.format(log_line),
            "max_tokens": 128,
            "temperature": 0.0,
            # CRITICAL UPGRADE: vLLM Server-Side Enforcement
            # This replaces Loom's prompt appending logic.
            "guided_json": json_schema
        })
        
        if resp.status_code == 200:
            raw_text = resp.json()['choices'][0]['text']
            
            # Trust but Verify
            try:
                actual_dict = json.loads(raw_text)
                
                # Scoring Logic
                match = True
                if actual_dict.get("src_ip") != expected_dict.get("src_ip"): match = False
                if actual_dict.get("user") != expected_dict.get("user"): match = False
                
                if match:
                    correct_count += 1
                else:
                    results.append({"error": "Mismatch", "expected": expected_dict, "actual": actual_dict})
            except json.JSONDecodeError:
                # This should be impossible with guided_json
                results.append({"error": "Invalid JSON", "raw": raw_text})
        else:
            print(f"Error {resp.status_code}: {resp.text}")

    except Exception as e:
        print(f"Connection Error: {e}")

accuracy = (correct_count / len(test_cases)) * 100
print(f"\nFinal Accuracy (Guided): {accuracy:.2f}%")

# DEBUG BLOCK
print("\n--- DEBUG: Top 3 Failures ---")
for i, res in enumerate(results[:3]):
    if not res.get("match", False):
        print(f"\nCase #{i+1}:")
        print(f"Expected: {res.get('expected')}")
        print(f"Actual (Raw): {res.get('actual')}")
        print(f"Error (if any): {res.get('error')}")
        print(f"Raw String from API: {res.get('raw')}")