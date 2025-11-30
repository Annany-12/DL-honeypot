"""Automated demo script for the AI-Driven Honeypot Simulator."""

import paramiko
import time
import threading
from typing import List
import random

class HoneypotDemoAttacker:
    """Simulates an attacker for demo purposes."""
    
    def __init__(self, target_host: str = "localhost", target_port: int = 2222):
        self.target_host = target_host
        self.target_port = target_port
        self.ssh_client = None
        self.channel = None
        
    def connect(self, username: str = "admin", password: str = "admin123") -> bool:
        """Connect to the honeypot via SSH."""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            print(f"🔌 Connecting to honeypot at {self.target_host}:{self.target_port}")
            self.ssh_client.connect(
                hostname=self.target_host,
                port=self.target_port,
                username=username,
                password=password,
                timeout=10
            )
            
            self.channel = self.ssh_client.invoke_shell()
            
            # Wait for initial prompt
            time.sleep(1)
            self._read_output()
            
            print("✅ Connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def _read_output(self, timeout: int = 2) -> str:
        """Read output from the SSH channel."""
        output = ""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.channel.recv_ready():
                data = self.channel.recv(1024).decode('utf-8', errors='ignore')
                output += data
                if data.endswith('$ '):
                    break
            time.sleep(0.1)
        
        return output
    
    def execute_command(self, command: str, wait_time: int = 3) -> str:
        """Execute a command and return the output."""
        try:
            print(f"💻 Executing: {command}")
            
            # Send command
            self.channel.send(command + '\n')
            
            # Wait and read output
            time.sleep(wait_time)
            output = self._read_output()
            
            print(f"📤 Output received ({len(output)} chars)")
            return output
            
        except Exception as e:
            print(f"❌ Command execution failed: {e}")
            return ""
    
    def run_reconnaissance_phase(self):
        """Phase 1: Reconnaissance commands."""
        print("\n🔍 PHASE 1: RECONNAISSANCE")
        print("-" * 40)
        
        commands = [
            "whoami",
            "hostname", 
            "uname -a",
            "id",
            "pwd",
            "ls -la",
            "ps aux",
            "netstat -an",
            "cat /etc/passwd"
        ]
        
        for cmd in commands:
            output = self.execute_command(cmd)
            time.sleep(random.uniform(1, 3))  # Realistic delay
    
    def run_exploration_phase(self):
        """Phase 2: File system exploration."""
        print("\n📁 PHASE 2: FILE SYSTEM EXPLORATION")
        print("-" * 40)
        
        commands = [
            "ls -la /home/admin",
            "cd Documents",
            "ls -la",
            "cat notes.txt",
            "cat passwords.txt",
            "cd ..",
            "ls -la .ssh",
            "cat .ssh/id_rsa",
            "cd /var/log",
            "ls -la",
            "tail -n 10 auth.log",
            "tail -n 20 syslog"
        ]
        
        for cmd in commands:
            output = self.execute_command(cmd)
            time.sleep(random.uniform(1, 4))
    
    def run_database_phase(self):
        """Phase 3: Database exploration."""
        print("\n🗄️ PHASE 3: DATABASE EXPLORATION")
        print("-" * 40)
        
        commands = [
            "mysql -e 'show databases;'",
            "mysql -e 'show tables;'",
            "mysql -e 'select * from customers limit 5;'",
            "mysql -e 'select * from transactions limit 10;'",
            "cd /var/lib/mysql",
            "ls -la",
            "cat customers.sql | head -20"
        ]
        
        for cmd in commands:
            output = self.execute_command(cmd)
            time.sleep(random.uniform(2, 5))
    
    def run_network_scanning_phase(self):
        """Phase 4: Network scanning."""
        print("\n🌐 PHASE 4: NETWORK SCANNING")
        print("-" * 40)
        
        commands = [
            "nmap localhost",
            "nmap -p 1-1000 localhost",
            "nmap -sV localhost",
            "netstat -tuln",
            "ss -tuln"
        ]
        
        for cmd in commands:
            output = self.execute_command(cmd, wait_time=5)  # Network commands take longer
            time.sleep(random.uniform(2, 4))
    
    def run_persistence_phase(self):
        """Phase 5: Persistence attempts."""
        print("\n🔒 PHASE 5: PERSISTENCE ATTEMPTS")
        print("-" * 40)
        
        commands = [
            "crontab -l",
            "cat /etc/crontab",
            "ls -la /etc/cron.d",
            "cat .bash_history",
            "history",
            "find / -name '*.conf' -type f 2>/dev/null | head -10",
            "grep -r 'password' /etc/ 2>/dev/null | head -5"
        ]
        
        for cmd in commands:
            output = self.execute_command(cmd)
            time.sleep(random.uniform(1, 3))
    
    def disconnect(self):
        """Disconnect from the honeypot."""
        try:
            if self.channel:
                self.channel.close()
            if self.ssh_client:
                self.ssh_client.close()
            print("🔌 Disconnected from honeypot")
        except Exception as e:
            print(f"⚠️ Disconnect error: {e}")

def run_automated_demo(duration_minutes: int = 10):
    """Run automated demo attack simulation."""
    print("🎭 STARTING AUTOMATED HONEYPOT DEMO")
    print("=" * 50)
    
    # Create attacker instance
    attacker = HoneypotDemoAttacker()
    
    # Connect to honeypot
    if not attacker.connect():
        print("❌ Demo failed - could not connect to honeypot")
        return
    
    try:
        # Run attack phases
        attacker.run_reconnaissance_phase()
        time.sleep(2)
        
        attacker.run_exploration_phase()
        time.sleep(2)
        
        attacker.run_database_phase()
        time.sleep(2)
        
        attacker.run_network_scanning_phase()
        time.sleep(2)
        
        attacker.run_persistence_phase()
        
        print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print("🎯 Check the dashboard for AI generation activities!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    finally:
        attacker.disconnect()

def interactive_demo():
    """Run interactive demo mode."""
    print("🎮 INTERACTIVE DEMO MODE")
    print("=" * 30)
    
    attacker = HoneypotDemoAttacker()
    
    if not attacker.connect():
        return
    
    try:
        while True:
            command = input("\n🔍 Enter command (or 'quit' to exit): ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            if command:
                output = attacker.execute_command(command)
                print(f"\n📤 Output:\n{output}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Interactive session ended")
    
    finally:
        attacker.disconnect()

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Honeypot Demo Script")
    parser.add_argument("--mode", choices=["auto", "interactive"], default="auto",
                       help="Demo mode: auto (automated) or interactive")
    parser.add_argument("--duration", type=int, default=10,
                       help="Duration in minutes for automated demo")
    parser.add_argument("--host", default="localhost",
                       help="Target honeypot host")
    parser.add_argument("--port", type=int, default=2222,
                       help="Target honeypot SSH port")
    
    args = parser.parse_args()
    
    if args.mode == "auto":
        run_automated_demo(args.duration)
    else:
        interactive_demo()

if __name__ == "__main__":
    main()
