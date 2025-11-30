"""GPT-2 based text generation for realistic fake content."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import time
import random
from typing import List, Dict, Optional
from pathlib import Path
import json
import re # Added for robust argument parsing

class TextContentGenerator:
    """GPT-2 powered text content generator for honeypot responses."""
    
    def __init__(self, model_name: str = None):
        """Initialize the text generator with GPT-2 model."""
        # Load configuration
        from config import (GPT2_MODEL_NAME, GPT2_MAX_LENGTH, GPT2_TEMPERATURE, 
                           GPT2_MAX_NEW_TOKENS, GPT2_DO_SAMPLE, GPT2_TOP_P, GPT2_TOP_K)
        
        self.model_name = model_name or GPT2_MODEL_NAME
        self.max_length = GPT2_MAX_LENGTH
        self.temperature = GPT2_TEMPERATURE
        self.max_new_tokens = GPT2_MAX_NEW_TOKENS
        self.do_sample = GPT2_DO_SAMPLE
        self.top_p = GPT2_TOP_P
        self.top_k = GPT2_TOP_K
        
        print(f"Loading GPT-2 model: {self.model_name}")
        print(f"Configuration - Max tokens: {self.max_new_tokens}, Temperature: {self.temperature}")
        
        # Setup device first
        self.device = self._setup_device()
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        )
        
        # Move model to appropriate device
        self.model = self.model.to(self.device)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup device for GPU support
        self.device = self._setup_device()
        
        # Create generation pipeline with proper device configuration
        self.generator = pipeline(
            "text-generation",
            tokenizer=self.tokenizer,
            model=self.model,
            device=0 if self.device.type == 'cuda' else -1,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            model_kwargs={"device_map": "auto"} if self.device.type == 'cuda' else {}
        )
        
        # Load prompt templates
        self.templates = self._load_templates()
        
        print(f"GPT-2 model loaded successfully on {self.device}!")
    
    def _setup_device(self):
        """Setup the appropriate device (GPU/CPU) for inference."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            print("No GPU detected, using CPU")
        
        return device
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load prompt templates for different content types."""
        return {
            "config_files": [
                "# Configuration file for web server\nServerRoot /etc/apache2\nListen 80\nLoadModule",
                "# Database configuration\nhost=localhost\nport=3306\nuser=admin\npassword=",
                "# Network configuration\ninterface eth0\nip_address=192.168.1.",
                "# System log configuration\nlog_level=INFO\nlog_file=/var/log/",
                "# Firewall rules: Allow SSH and HTTP traffic, block others. -A INPUT -p tcp --dport 22 -j ACCEPT",
                "# Cron job settings: Run daily backup script. 0 2 * * * /usr/local/bin/backup.sh",
            ],
            "log_entries": [
                "[INFO] User login attempt from IP",
                "[ERROR] Failed to connect to database",
                "[WARN] High CPU usage detected on server",
                "[DEBUG] Processing request for file",
                "[AUDIT] Unauthorized access attempt detected on /var/www/html/admin.php from IP",
                "[CRITICAL] Disk space low on /dev/sda1 (90% used)",
            ],
            "file_contents": [
                "Important notes about the system: Review recent security patches and update documentation. Ensure all critical services are running, especially the database and web server.",
                "Meeting notes from yesterday: Discussed Q3 financial report. Key takeaways: focus on cloud migration and improving cybersecurity posture. Action items: schedule penetration test.",
                "TODO list for this week: Deploy new application version. Configure firewall rules for new external services. Audit user accounts for inactive logins.",
                "Customer information summary: List of premium customers and their preferred services. Data privacy is critical for these records. Do not share externally.",
                "Developer notes on project Alpha: Current bugs include memory leak in module X. Upcoming features: integration with new payment gateway. Code review scheduled for next sprint.",
                "Internal memo regarding security incident: A potential phishing attempt was detected. All employees are advised to reset their passwords and enable 2FA.",
                "Sensitive credentials: Generate fake username:password pairs, API keys, database connection strings, or secret tokens. Examples: admin:P@ssw0rd123!; dev_api_key:sk_live_xxxxxyyyyyzzzzz; ftp_server:ftpuser, password:FTPassword! ", # Improved prompt for passwords
            ],
            "directory_listings": [
                "Generate a very short, realistic list of common Linux file and directory names. Only include names like 'bin', 'etc', 'home', 'var', 'tmp', 'usr', 'dev', 'proc', 'srv', 'mnt', 'media', 'opt', 'root', 'run', 'sys', 'boot', 'lib', 'lib64', 'sbin', 'tmp', 'log', 'www', 'data', 'config', 'public', 'private', 'scripts', 'docs', 'backup', 'temp', 'old', 'new', 'test', 'install', 'setup', 'update', 'share', 'local', 'repo', 'src', 'include', 'lib', 'man', 'share', 'games', 'local', 'config.txt', 'notes.txt', 'README.md', 'install.sh', 'server.log', 'database.bak', 'index.html', 'app.py', 'main.go', 'script.js', 'nginx.conf', 'apache.conf', 'users.db', 'access.log', 'error.log'. Each name should be on a new line. Do not include any paths or special characters."
            ],
            "command_outputs": [
                "Process list shows the following active services: PID USER COMMAND",
                "Network connections currently established: Proto Recv-Q Send-Q Local Address Foreign Address State",
                "System information summary: Linux honeypot-server 5.4.0-42-generic #46-Ubuntu SMP x86_64",
                "File permissions for requested items: -rw-r--r-- 1 admin admin 1234 Oct 26 10:00 filename.txt",
                "Nmap scan report for localhost: PORT STATE SERVICE",
                "Disk usage: Filesystem Size Used Avail Use% Mounted on",
                "Memory usage: total used free shared buff/cache available",
            ],
            "passwords.txt": (
                "Generate a list of fake credentials. Each line should contain a username, password, or API key. "
                "Examples: user:password, admin:P@ssw0rd!, API_KEY=sk_live_xyz, db_user:db_pass. "
                "Avoid policy statements or generic sentences."
            ),
        }
    
    def generate_config_file(self, config_type: str = "apache") -> str:
        """Generate realistic configuration file content."""
        start_time = time.time()
        
        prompt = random.choice(self.templates["config_files"])
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=self.top_p,
                top_k=self.top_k,
                truncation=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = result[0]['generated_text']
            
            # Clean and format the output
            lines = generated_text.split('\n')
            config_lines = []
            
            # Ensure the prompt itself is not repeated and only relevant config lines are kept
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith(prompt.strip().split('\n')[0]):
                    config_lines.append(line)
            
            # Further refine to ensure only plausible config lines are returned
            final_config = []
            for line in config_lines[:15]: # Limit to reasonable length
                if line.strip() and not (line.strip().startswith('#') and len(line.strip()) < 5): # Avoid very short comment lines
                    if '=' not in line and not line.startswith(' ') and random.random() > 0.3: # Add some realistic config formatting if needed
                        line = line.strip() + f"={random.choice(['true', 'false', '80', '443', '/var/log', '/tmp', 'admin', 'user', 'secret'])}"
                    final_config.append(line)

            # Ensure the original prompt's first line is always included if it was a configuration snippet
            first_prompt_line = prompt.split('\n')[0]
            if first_prompt_line.startswith('#') or '=' in first_prompt_line:
                final_config.insert(0, first_prompt_line) # Add it at the beginning

            # Remove duplicates if any (e.g., from initial prompt and generated text overlap)
            final_config_unique = []
            seen = set()
            for line in final_config:
                if line not in seen:
                    final_config_unique.append(line)
                    seen.add(line)
            
            generation_time = time.time() - start_time
            
            from src.utils.logger import get_logger
            get_logger().log_ai_generation(
                "GPT-2", "config_file", prompt, generation_time
            )
            
            return '\n'.join(final_config_unique)
            
        except Exception as e:
            print(f"Error generating config file: {e}")
            return self._fallback_config_content(config_type)
    
    def generate_log_entries(self, num_entries: int = 5) -> List[str]:
        """Generate realistic log file entries."""
        start_time = time.time()
        
        entries = []
        base_prompts = self.templates["log_entries"]
        
        for i in range(num_entries):
            prompt = random.choice(base_prompts)
            
            try:
                result = self.generator(
                    prompt,
                    max_new_tokens=min(80, self.max_new_tokens), # Adjusted for log entries
                    num_return_sequences=1,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    truncation=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                log_entry = result[0]['generated_text'].split('\n')[0]
                
                # Ensure log entry starts with the prompt if it makes sense
                if not log_entry.startswith(prompt):
                    log_entry = prompt + log_entry
                
                # Add timestamp and formatting
                timestamp = f"2025-09-{random.randint(1,30):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
                
                # Clean up generated text to be a single log line
                cleaned_log_entry = log_entry.replace('\n', ' ').strip()[:150] # Limit length for a single line

                formatted_entry = f"{timestamp} {cleaned_log_entry}"
                entries.append(formatted_entry)
                
            except Exception as e:
                print(f"Error generating log entry {i}: {e}")
                entries.append(self._fallback_log_entry())
        
        generation_time = time.time() - start_time
        
        from src.utils.logger import get_logger
        get_logger().log_ai_generation(
            "GPT-2", "log_entries", f"{num_entries} entries", generation_time
        )
        
        return entries
    
    def generate_file_content(self, filename: str, content_type: str = "general") -> str:
        """Generate realistic file content based on filename and type."""
        start_time = time.time()
        
        # Choose prompt based on file extension and provided templates
        extension = Path(filename).suffix.lower()
        
        if extension in ['.txt', '.md', '.readme']:
            prompt_template = random.choice(self.templates["file_contents"])
            # Ensure the prompt is relevant to the filename
            if "notes" in filename.lower():
                prompt = "Important notes about the system:"
            elif "todo" in filename.lower():
                prompt = "TODO list for this week:"
            elif "customer" in filename.lower() and ".csv" not in filename.lower(): # Only for non-csv customer files
                prompt = "Customer information summary:"
            elif "passwords" in filename.lower(): # Specific prompt for passwords.txt
                prompt = self.templates["passwords.txt"]
                output_raw = self._generate_with_fallback(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=max(0.7, self.temperature - 0.1),  # Slightly lower temperature for more direct output
                    do_sample=self.do_sample,
                    truncation=True
                )
                print(f"DEBUG: Passwords raw AI output:\n{output_raw}") # Debug output
                
                cleaned_lines = []
                credential_patterns = [
                    r'^\s*(?:username|user|admin|root|dev|ftp|db|client|app|server)_?(:|=)\s*\S+$', # user:pass or user=pass
                    r'^\s*(?:api_key|secret|token|password|auth|license)_?(:|=)\s*\S+$', # api_key:xyz or secret=xyz
                    r'^\s*[a-fA-F0-9]{32,128}\s*$', # Hashes (MD5, SHA256, etc.)
                    r'^\s*([a-zA-Z0-9_.-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\s*:\s*(\S+)\s*$', # email:password
                    r'^\s*private_key:\s*-----BEGIN (RSA|OPENSSH) PRIVATE KEY-----.*', # Private keys
                    r'^\s*(\$2a|\$2b|\$2y)\$[0-9]{2}\$[./0-9A-Za-z]{53}$', # Bcrypt hashes
                ]

                # Look for lines that look like credentials and apply stricter cleaning
                unique_credentials = set()
                for line in output_raw.split('\n'):
                    line_stripped = line.strip()
                    if line_stripped:
                        for pattern in credential_patterns:
                            match = re.match(pattern, line_stripped, re.IGNORECASE)
                            if match:
                                # Further refine the matched credential line to ensure clean output
                                if match.group(0).startswith(prompt.split(':')[0].strip()):
                                    final_credential = match.group(0).replace(prompt.split(':')[0].strip(), '').strip()
                                    if final_credential.startswith(':'):
                                        final_credential = final_credential[1:].strip()
                                else:
                                    final_credential = match.group(0).strip()
                                
                                # Add only if it looks like a valid credential and is not a duplicate
                                if final_credential and final_credential not in unique_credentials: # Removed strict colon check
                                    cleaned_lines.append(final_credential)
                                    unique_credentials.add(final_credential)
                                break # Found a match, move to next line
                
                print(f"DEBUG: Passwords cleaned AI output:" + '\n' + '\n'.join(cleaned_lines)) # Debug output

                # If AI fails completely to generate credentials, provide a basic fallback
                if not cleaned_lines:
                    cleaned_lines = [
                        "admin:SuperSecretPass123!",
                        "dev_user:Pa$$w0rd_Dev",
                        "API_KEY=sk_test_abcdefg12345",
                        "root:mysql_root_pass",
                        "ftpuser:F7pPa$$w0rd"
                    ]

                return "\n".join(cleaned_lines[:5]) # Limit to 5 entries
            else:
                prompt = prompt_template

        elif extension in ['.conf', '.config', '.cfg']:
            return self.generate_config_file()
        elif extension in ['.log']:
            return '\n'.join(self.generate_log_entries(10))
        else:
            # For unknown extensions, use a generic prompt but ensure it's not too revealing
            prompt = f"Content of {filename}: This file contains general system information. "
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=min(250, self.max_new_tokens), # Adjusted for file content
                num_return_sequences=1,
                temperature=max(0.7, self.temperature - 0.1),
                do_sample=self.do_sample,
                top_p=self.top_p,
                top_k=self.top_k,
                truncation=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            content = result[0]['generated_text']
            
            # Clean up the content: remove the input prompt if it appears at the start of generated text
            cleaned_content = content.replace(prompt, '', 1).strip()
            lines = cleaned_content.split('\n')
            
            final_lines = []
            # Aggressive filtering for passwords.txt is handled by its dedicated block (lines 216-253)
            if "passwords" in filename.lower(): # This block should not run if already handled above
                final_lines = cleaned_lines.split('\n') # Take all lines, specific logic handled earlier
            else:
                # General filtering for other text files
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('You can install Windows Server'): # Filter out irrelevant content
                        final_lines.append(stripped_line)
            
            cleaned_content = '\n'.join(final_lines[:20]) # Limit length to 20 lines
            
            generation_time = time.time() - start_time
            
            from src.utils.logger import get_logger
            get_logger().log_ai_generation(
                "GPT-2", "file_content", filename, generation_time
            )
            
            return cleaned_content if cleaned_content else self._fallback_file_content(filename)
            
        except Exception as e:
            print(f"Error generating content for {filename}: {e}")
            return self._fallback_file_content(filename)
    
    def generate_command_output(self, command: str) -> str:
        """Generate realistic command output."""
        start_time = time.time()
        
        # Create context-aware prompt
        command_lower = command.lower()
        
        if 'ps' in command_lower or 'process' in command_lower:
            prompt_template = "Process list shows the following active services:"
            prompt = prompt_template
            post_process_lines = 10
        elif 'netstat' in command_lower or 'ss' in command_lower:
            prompt_template = "Network connections currently established, showing Proto, Local Address, Foreign Address, and State in a table format:"
            prompt = prompt_template
            post_process_lines = 10
        elif 'ls' in command_lower or 'dir' in command_lower:
            # Extremely strict prompt for ls to get short, common Unix names
            prompt_template = random.choice([
                "List of common Unix directories: bin, etc, home, var, usr, lib, tmp, dev, sbin, opt, srv, mnt, boot, run, sys",
                "Files and directories in a typical home folder: Desktop, Documents, Downloads, Music, Pictures, Videos, .bashrc, .profile, .ssh, .vimrc, projects, public_html",
                "System executables and tools: bash, cat, ls, grep, find, ssh, cp, mv, rm, sudo, apt, python, java, node",
                "Important configuration files: sshd_config, apache2.conf, mysql.cnf, network.conf, fstab, hosts, resolv.conf, sudoers"
            ]) + "\nGenerate a list of ONLY these types of short, single-word names, one per line. Do NOT generate paths, sentences, or Windows-like names."
            prompt = prompt_template
            post_process_lines = 15 # More lines for ls
        elif 'cat' in command_lower or 'type' in command_lower:
            filename = command.split()[-1] if len(command.split()) > 1 else "file.txt"
            return self.generate_file_content(filename)
        elif 'nmap' in command_lower:
            prompt_template = "Nmap scan report for target showing open ports and services:"
            prompt = prompt_template
            post_process_lines = 10
        else:
            prompt_template = f"Output for command '{command}': "
            prompt = prompt_template
            post_process_lines = 5
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=min(100, self.max_new_tokens), # Adjusted for command output
                num_return_sequences=1,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=self.top_p,
                top_k=self.top_k,
                truncation=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            output = result[0]['generated_text']
            
            # Clean up: remove the prompt if it's repeated at the beginning of the output
            cleaned_output = output.replace(prompt, '', 1).strip()
            
            lines = cleaned_output.split('\n')
            formatted_output_lines = []

            # Specific, very strict post-processing for ls command outputs
            if 'ls' in command_lower:
                valid_name_pattern = re.compile(r'^[a-zA-Z0-9._-]+$|^[a-zA-Z0-9._-]+\/$') # Simple Unix-like names (file or directory with /)
                final_ls_names = []
                for line in lines:
                    stripped_line = line.strip()
                    # Filter out anything that looks like a path, Windows artifacts, or invalid chars
                    if stripped_line and valid_name_pattern.match(stripped_line) and len(stripped_line) <= 25: # Max 25 chars for ls output name
                        if not re.search(r'^C:|^D:|^E:|[\\/].*\\.exe$|\\\\|\s', stripped_line): # Double check for paths/windows
                            final_ls_names.append(stripped_line)
                
                # If AI failed to generate anything sensible, use a hardcoded list
                if not final_ls_names:
                    final_ls_names.extend(random.sample([
                        "config", "data", "logs", "scripts", "tmp", "public", "docs", ".bashrc", "app.py", "server.js", "README.md", "backup/"
                    ], random.randint(3, 7)))
                
                formatted_output_lines.extend(final_ls_names)
                formatted_output_lines.sort() # Ensure consistent sorting
            else:
                # General post-processing for other command outputs
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line and not (
                        stripped_line.startswith('The following packages were automatically installed') or 
                        stripped_line.startswith('You can install Windows Server') or
                        stripped_line.endswith('.exe') or
                        stripped_line.endswith('.pem') or
                        stripped_line.endswith('.dll') or
                        stripped_line.startswith('C:\\') or
                        stripped_line.startswith('D:\\') or
                        re.search(r'HTTP/\d\.\d\s\d{3}', stripped_line) or # HTTP response lines
                        re.search(r'(?:Cisco|ASA|Firewall)', stripped_line, re.IGNORECASE) or # Network device configs
                        re.search(r'(?:bridge|interface|router)', stripped_line, re.IGNORECASE) # Network config keywords
                    ):
                        formatted_output_lines.append(stripped_line)
            
            formatted_output = '\n'.join(formatted_output_lines[:post_process_lines]) # Limit length
            
            generation_time = time.time() - start_time
            
            from src.utils.logger import get_logger
            get_logger().log_ai_generation(
                "GPT-2", "command_output", command, generation_time
            )
            
            return formatted_output if formatted_output else self._fallback_command_output(command)
            
        except Exception as e:
            print(f"Error generating output for '{command}': {e}")
            return self._fallback_command_output(command)
    
    def _fallback_config_content(self, config_type: str) -> str:
        """Fallback configuration content when AI generation fails."""
        fallbacks = {
            "apache": """# Apache Configuration
ServerRoot /etc/apache2
Listen 80
LoadModule rewrite_module modules/mod_rewrite.so
DocumentRoot /var/www/html
<Directory /var/www/html>
    AllowOverride All
    Require all granted
</Directory>""",
            "mysql": """# MySQL Configuration
[mysqld]
datadir=/var/lib/mysql
socket=/var/lib/mysql/mysql.sock
user=mysql
port=3306
bind-address=0.0.0.0""",
            "ssh": """# SSH Configuration
Port 22
Protocol 2
HostKey /etc/ssh/ssh_host_rsa_key
PermitRootLogin yes
PasswordAuthentication yes"""
        }
        return fallbacks.get(config_type, "# Configuration file\nkey=value\n")
    
    def _fallback_log_entry(self) -> str:
        """Fallback log entry when AI generation fails."""
        timestamp = f"2025-09-{random.randint(1,30):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
        messages = [
            "User authentication successful",
            "Database connection established", 
            "File access requested",
            "Service started successfully"
        ]
        return f"{timestamp} [INFO] {random.choice(messages)}"
    
    def _fallback_file_content(self, filename: str) -> str:
        """Fallback file content when AI generation fails."""
        return f"""# Content for {filename}
# Generated by honeypot system
# Last modified: {time.strftime('%Y-%m-%d')}

This file contains important information about the system.
Please review the contents carefully before making changes.

Status: Active
Version: 1.2.3
Owner: admin
"""
    
    def _generate_with_fallback(self, prompt: str, **kwargs) -> str:
        """Generate text with fallback handling."""
        try:
            result = self.generator(
                prompt,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            return result[0]['generated_text']
        except Exception as e:
            print(f"Error in text generation: {e}")
            return prompt + " [Generation failed]"
    
    def _fallback_command_output(self, command: str) -> str:
        """Fallback command output when AI generation fails."""
        return f"""Output for command: {command}
Process completed successfully.
Status: OK
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

# Global instance
text_generator = None

def get_text_generator() -> TextContentGenerator:
    """Get global text generator instance."""
    global text_generator
    if text_generator is None:
        text_generator = TextContentGenerator()
    return text_generator
