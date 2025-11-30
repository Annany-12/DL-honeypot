"""Generate sample datasets for training AI models."""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import json

def generate_network_logs(num_entries: int = 10000) -> pd.DataFrame:
    """Generate sample network log data."""
    np.random.seed(42)
    
    # IP addresses
    ips = [f"192.168.1.{random.randint(1, 254)}" for _ in range(100)]
    external_ips = [f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(50)]
    
    # Services and ports
    services = ["ssh", "http", "https", "ftp", "smtp", "dns", "mysql", "redis"]
    ports = [22, 80, 443, 21, 25, 53, 3306, 6379]
    
    # Log levels
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    level_weights = [0.6, 0.2, 0.1, 0.1]
    
    data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_entries):
        timestamp = base_time + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        level = random.choices(levels, weights=level_weights)[0]
        source_ip = random.choice(ips + external_ips)
        dest_ip = random.choice(ips)
        service = random.choice(services)
        port = random.choice(ports)
        
        # Generate realistic messages based on level
        if level == "ERROR":
            messages = [
                f"Connection refused from {source_ip}",
                f"Authentication failed for {source_ip}",
                f"Service {service} failed to start",
                f"Database connection lost"
            ]
        elif level == "WARN":
            messages = [
                f"High CPU usage detected",
                f"Memory usage above threshold",
                f"Unusual traffic pattern from {source_ip}"
            ]
        else:
            messages = [
                f"Connection established from {source_ip}:{port}",
                f"Service {service} started successfully",
                f"User authenticated from {source_ip}",
                f"Request processed for {dest_ip}"
            ]
        
        message = random.choice(messages)
        
        data.append({
            "timestamp": timestamp.isoformat(),
            "level": level,
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "service": service,
            "port": port,
            "message": message,
            "bytes_sent": random.randint(100, 10000),
            "bytes_received": random.randint(100, 5000),
            "duration": random.uniform(0.1, 30.0)
        })
    
    return pd.DataFrame(data)

def generate_user_behavior_data(num_users: int = 1000) -> pd.DataFrame:
    """Generate user behavior data for training."""
    np.random.seed(42)
    
    # User types
    user_types = ["admin", "user", "guest", "service"]
    
    # Common commands by user type
    admin_commands = ["sudo", "systemctl", "vim", "tail", "grep", "ps", "netstat", "mysql"]
    user_commands = ["ls", "cd", "cat", "less", "find", "wget", "curl", "python"]
    guest_commands = ["ls", "pwd", "whoami", "cat", "help"]
    service_commands = ["curl", "wget", "python", "java", "node"]
    
    data = []
    
    for user_id in range(1, num_users + 1):
        user_type = random.choices(
            user_types, 
            weights=[0.05, 0.7, 0.15, 0.1]
        )[0]
        
        # Number of sessions for this user
        num_sessions = random.randint(1, 20)
        
        for session in range(num_sessions):
            session_start = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Commands in this session
            if user_type == "admin":
                commands = random.choices(admin_commands, k=random.randint(5, 20))
            elif user_type == "user":
                commands = random.choices(user_commands, k=random.randint(3, 15))
            elif user_type == "guest":
                commands = random.choices(guest_commands, k=random.randint(2, 8))
            else:  # service
                commands = random.choices(service_commands, k=random.randint(1, 5))
            
            for i, command in enumerate(commands):
                command_time = session_start + timedelta(minutes=i * random.randint(1, 5))
                
                data.append({
                    "user_id": user_id,
                    "user_type": user_type,
                    "session_id": f"{user_id}_{session}",
                    "timestamp": command_time.isoformat(),
                    "command": command,
                    "command_order": i + 1,
                    "session_duration": len(commands) * 3,  # Rough estimate
                    "success": random.choices([True, False], weights=[0.9, 0.1])[0]
                })
    
    return pd.DataFrame(data)

def generate_file_access_patterns(num_entries: int = 5000) -> pd.DataFrame:
    """Generate file access pattern data."""
    np.random.seed(42)
    
    # File types and locations
    system_files = [
        "/etc/passwd", "/etc/shadow", "/etc/hosts", "/etc/crontab",
        "/var/log/auth.log", "/var/log/syslog", "/var/log/messages"
    ]
    
    user_files = [
        "/home/user1/documents/file.txt", "/home/user1/.bash_history",
        "/home/user1/.ssh/id_rsa", "/home/user1/downloads/data.csv"
    ]
    
    application_files = [
        "/opt/app/config.conf", "/opt/app/data.db", "/opt/app/logs/error.log"
    ]
    
    all_files = system_files + user_files + application_files
    
    access_types = ["read", "write", "execute", "delete"]
    access_weights = [0.6, 0.25, 0.1, 0.05]
    
    data = []
    
    for i in range(num_entries):
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        file_path = random.choice(all_files)
        access_type = random.choices(access_types, weights=access_weights)[0]
        user_id = random.randint(1, 100)
        
        # File size (bytes)
        if "log" in file_path:
            file_size = random.randint(1000, 1000000)
        elif "config" in file_path or "passwd" in file_path:
            file_size = random.randint(100, 10000)
        else:
            file_size = random.randint(500, 50000)
        
        data.append({
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "file_path": file_path,
            "access_type": access_type,
            "file_size": file_size,
            "success": random.choices([True, False], weights=[0.95, 0.05])[0],
            "duration": random.uniform(0.01, 2.0)
        })
    
    return pd.DataFrame(data)

def generate_attack_signatures(num_entries: int = 2000) -> pd.DataFrame:
    """Generate attack signature data for training."""
    np.random.seed(42)
    
    # Attack types
    attack_types = [
        "brute_force", "sql_injection", "command_injection", 
        "directory_traversal", "xss", "port_scan", "privilege_escalation"
    ]
    
    # Attack patterns
    attack_patterns = {
        "brute_force": [
            "multiple failed login attempts",
            "rapid password attempts",
            "dictionary attack pattern"
        ],
        "sql_injection": [
            "SELECT * FROM users WHERE",
            "UNION SELECT NULL",
            "' OR '1'='1"
        ],
        "command_injection": [
            "; cat /etc/passwd",
            "| whoami",
            "&& ls -la"
        ],
        "directory_traversal": [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "....//....//etc//passwd"
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>"
        ],
        "port_scan": [
            "rapid port probing",
            "sequential port access",
            "service enumeration"
        ],
        "privilege_escalation": [
            "sudo privilege abuse",
            "SUID binary exploitation",
            "kernel exploit attempt"
        ]
    }
    
    severity_levels = ["low", "medium", "high", "critical"]
    
    data = []
    
    for i in range(num_entries):
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        attack_type = random.choice(attack_types)
        pattern = random.choice(attack_patterns[attack_type])
        source_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        
        # Assign severity based on attack type
        if attack_type in ["sql_injection", "command_injection", "privilege_escalation"]:
            severity = random.choices(severity_levels, weights=[0.1, 0.2, 0.4, 0.3])[0]
        elif attack_type in ["brute_force", "port_scan"]:
            severity = random.choices(severity_levels, weights=[0.2, 0.5, 0.25, 0.05])[0]
        else:
            severity = random.choices(severity_levels, weights=[0.3, 0.4, 0.25, 0.05])[0]
        
        data.append({
            "timestamp": timestamp.isoformat(),
            "attack_type": attack_type,
            "pattern": pattern,
            "source_ip": source_ip,
            "target_ip": f"192.168.1.{random.randint(1, 254)}",
            "severity": severity,
            "blocked": random.choices([True, False], weights=[0.7, 0.3])[0],
            "confidence": random.uniform(0.5, 1.0)
        })
    
    return pd.DataFrame(data)

def main():
    """Generate all sample datasets."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    print("📊 Generating sample datasets for AI training...")
    
    # Generate datasets
    datasets = {
        "network_logs.csv": generate_network_logs(10000),
        "user_behavior.csv": generate_user_behavior_data(1000),
        "file_access_patterns.csv": generate_file_access_patterns(5000),
        "attack_signatures.csv": generate_attack_signatures(2000)
    }
    
    # Save datasets
    for filename, df in datasets.items():
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"✅ Generated {filename}: {len(df)} rows, {len(df.columns)} columns")
    
    # Generate metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "datasets": {
            name: {
                "rows": len(df),
                "columns": list(df.columns),
                "file_size_kb": filepath.stat().st_size // 1024 if filepath.exists() else 0
            }
            for name, df in datasets.items()
        }
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ All datasets generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("🎯 These datasets will be used to train the AI models for realistic content generation.")

if __name__ == "__main__":
    main()
