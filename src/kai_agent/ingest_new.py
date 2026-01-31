import asyncio
import httpx
import json
from pathlib import Path
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
# Since we are running ON the Spark, we can use localhost
VLLM_URL = "http://localhost:8005/v1/completions"
MODEL_NAME = "kai-log-parser"
LOG_FILE = Path("data/system.log")

# --- SCHEMA (Must match what we used in eval) ---
class SyslogEntry(BaseModel):
    timestamp: str = Field(description="e.g. 'Jan 28 12:00:00'")
    hostname: str
    process: str
    pid: int
    event_type: str
    user: str
    src_ip: str
    port: int

# One-Shot Prompt Template (The "Fix" from Day 9)
alpaca_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Parse the following syslog entry into structured JSON.
Example Input: Jan 28 12:00:00 firewall-1 sshd[1234]: Failed password for invalid user admin from 1.2.3.4 port 5555 ssh2
Example Response: {{"timestamp": "Jan 28 12:00:00", "hostname": "firewall-1", "process": "sshd", "pid": 1234, "event_type": "authentication_failure", "user": "admin", "src_ip": "1.2.3.4", "port": 5555}}

### Input:
{}

### Response:
"""

async def parse_log_line(client: httpx.AsyncClient, line: str):
    if not line.strip():
        return None

    # Generate schema for Guided Decoding
    json_schema = SyslogEntry.model_json_schema()
    
    try:
        response = await client.post(
            VLLM_URL,
            json={
                "model": MODEL_NAME,
                "prompt": alpaca_template.format(line),
                "max_tokens": 200,
                "temperature": 0.0,
                # The "Magic Switch" for 100% JSON reliability
                "guided_json": json_schema 
            },
            timeout=30.0
        )
        response.raise_for_status()
        result_text = response.json()['choices'][0]['text']
        return json.loads(result_text)
        
    except Exception as e:
        print(f"Error parsing line: {e}")
        return None

async def main():
    if not LOG_FILE.exists():
        print(f"Error: {LOG_FILE} not found!")
        return

    print(f"Reading logs from {LOG_FILE}...")
    
    # Process line by line
    async with httpx.AsyncClient() as client:
        with open(LOG_FILE, "r") as f:
            count = 0
            for line in f:
                # Skip empty lines
                if not line.strip(): continue
                
                parsed_data = await parse_log_line(client, line.strip())
                
                if parsed_data:
                    count += 1
                    # Pretty print the success
                    print(f"✅ [{count}] Parsed: {parsed_data['timestamp']} | User: {parsed_data['user']} | IP: {parsed_data['src_ip']}")
                    # Optional: Print full JSON for the first few to verify
                    if count <= 3:
                        print(f"   └─ {json.dumps(parsed_data)}")

if __name__ == "__main__":
    asyncio.run(main())