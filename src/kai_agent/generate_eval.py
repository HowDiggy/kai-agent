import json
import random

def generate_test_cases(num_cases=20):
    """
    Generates synthetic logs and their expected JSON output.
    """
    users = ["root", "admin", "paulo", "guest", "deploy"]
    ips = ["192.168.1.5", "10.0.0.200", "172.16.0.55", "2001:db8::1"] # Added IPv6 to be tricky
    processes = ["sshd", "kernel", "opnsense", "dhcpd"]
    
    test_cases = []
    
    for _ in range(num_cases):
        user = random.choice(users)
        ip = random.choice(ips)
        pid = random.randint(1000, 9999)
        port = random.randint(1024, 65535)
        
        # Scenario 1: SSH Failure
        log = f"Jan 28 12:00:00 firewall-1 sshd[{pid}]: Failed password for invalid user {user} from {ip} port {port} ssh2"
        expected = {
            "timestamp": "Jan 28 12:00:00",
            "hostname": "firewall-1",
            "process": "sshd",
            "pid": pid,
            "event_type": "authentication_failure",
            "user": user,
            "src_ip": ip,
            "port": port
        }
        
        test_cases.append({
            "input": log,
            "actual_output": None, # Filled in by model later
            "expected_output": json.dumps(expected)
        })
        
    return test_cases

if __name__ == "__main__":
    data = generate_test_cases()
    with open("data/eval_dataset.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} test cases in data/eval_dataset.json")