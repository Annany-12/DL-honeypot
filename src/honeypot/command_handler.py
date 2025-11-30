"""Command handling system for the fake shell environment."""

import os
import random
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from src.utils.logger import get_logger
import re # Added for regex in _handle_ls


class FakeFilesystem:
    """Simulate a realistic Linux filesystem structure."""
    
    def __init__(self):
        """Initialize the fake filesystem."""
        self.current_dir = "/home/admin"
        self.filesystem = self._create_filesystem_structure()
        self.file_contents = {}  # Cache for generated file contents
    
    def _create_filesystem_structure(self) -> Dict[str, Any]:
        """Create a realistic directory structure."""
        return {
            "/": {
                "type": "directory",
                "contents": {
                    "home": {
                        "type": "directory", 
                        "contents": {
                            "admin": {
                                "type": "directory",
                                "contents": {
                                    "Documents": {"type": "directory", "contents": {
                                        "notes.txt": {"type": "file", "size": 1024},
                                        "passwords.txt": {"type": "file", "size": 256},
                                        "customer_data.csv": {"type": "file", "size": 50000}
                                    }},
                                    "Downloads": {"type": "directory", "contents": {}},
                                    ".ssh": {"type": "directory", "contents": {
                                        "id_rsa": {"type": "file", "size": 1679},
                                        "id_rsa.pub": {"type": "file", "size": 394},
                                        "known_hosts": {"type": "file", "size": 2048}
                                    }},
                                    ".bash_history": {"type": "file", "size": 15000}
                                }
                            }
                        }
                    },
                    "var": {
                        "type": "directory",
                        "contents": {
                            "log": {
                                "type": "directory", 
                                "contents": {
                                    "auth.log": {"type": "file", "size": 100000},
                                    "syslog": {"type": "file", "size": 250000},
                                    "apache2": {"type": "directory", "contents": {
                                        "access.log": {"type": "file", "size": 500000},
                                        "error.log": {"type": "file", "size": 75000}
                                    }}
                                }
                            },
                            "lib": {
                                "type": "directory",
                                "contents": {
                                    "mysql": {"type": "directory", "contents": {
                                        "customers.sql": {"type": "file", "size": 2000000}
                                    }}
                                }
                            }
                        }
                    },
                    "etc": {
                        "type": "directory",
                        "contents": {
                            "passwd": {"type": "file", "size": 2048},
                            "shadow": {"type": "file", "size": 1024},
                            "ssh": {"type": "directory", "contents": {
                                "sshd_config": {"type": "file", "size": 3264}
                            }},
                            "apache2": {"type": "directory", "contents": {
                                "apache2.conf": {"type": "file", "size": 7224}
                            }}
                        }
                    },
                    "usr": {
                        "type": "directory",
                        "contents": {
                            "bin": {"type": "directory", "contents": {}},
                            "local": {"type": "directory", "contents": {}}
                        }
                    }
                }
            }
        }
    
    def navigate_to_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Navigate to a specific path in the filesystem."""
        if path.startswith("/"):
            # Absolute path
            parts = [p for p in path.split("/") if p]
            current = self.filesystem["/"]
        else:
            # Relative path
            parts = [p for p in path.split("/") if p]
            current = self.navigate_to_path(self.current_dir)
            if current is None:
                return None
        
        for part in parts:
            if part == ".":
                continue
            elif part == "..":
                # Handle going up - simplified for demo
                continue
            elif current["type"] == "directory" and part in current["contents"]:
                current = current["contents"][part]
            else:
                return None
        
        return current
    
    def list_directory(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """List contents of a directory."""
        if path is None:
            path = self.current_dir
        
        target = self.navigate_to_path(path)
        if target is None or target["type"] != "directory":
            return []
        
        items = []
        for name, info in target["contents"].items():
            items.append({
                "name": name,
                "type": info["type"],
                "size": info.get("size", 0),
                "permissions": "drwxr-xr-x" if info["type"] == "directory" else "-rw-r--r--"
            })
        
        return items
    
    def file_exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        return self.navigate_to_path(path) is not None
    
    def get_file_info(self, path: str) -> Optional[Dict[str, Any]]:
        """Get information about a file."""
        return self.navigate_to_path(path)

class CommandHandler:
    """Handle shell commands and generate appropriate responses."""
    
    def __init__(self):
        """Initialize the command handler."""
        self.filesystem = FakeFilesystem()
        self.command_history = []
        self.environment_vars = {
            "HOME": "/home/admin",
            "USER": "admin",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PWD": "/home/admin",
            "SHELL": "/bin/bash"
        }
        
        # Static command responses for common utilities
        self.static_responses = self._load_static_responses()
    
    def _load_static_responses(self) -> Dict[str, Any]:
        """Load static responses for common commands."""
        return {
            "whoami": "admin",
            "hostname": "honeypot-server",
            "uptime": " 15:42:13 up 7 days, 12:34,  2 users,  load average: 0.23, 0.45, 0.12",
            "uname -a": "Linux honeypot-server 5.4.0-42-generic #46-Ubuntu SMP Fri Jul 10 00:24:02 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux",
            "id": "uid=1000(admin) gid=1000(admin) groups=1000(admin),4(adm),24(cdrom),27(sudo),30(dip),46(plugdev),120(lpadmin),131(lxd),132(sambashare)",
            "date": time.strftime("%a %b %d %H:%M:%S %Z %Y"),
            "pwd": "/home/admin"
        }
    
    def handle_command(self, command: str, client_ip: str) -> str:
        """Process a command and return appropriate response."""
        command = command.strip()
        if not command:
            return ""
        
        # Log the command
        from src.utils.logger import get_logger
        logger = get_logger()
        logger.log_command(client_ip, command, "processing")
        
        # Add to history
        self.command_history.append(command)
        
        # Parse command - more robustly handle quoted arguments and -e flag
        # Use shlex to handle shell-like parsing, including quotes
        import shlex
        try:
            split_command = shlex.split(command)
            cmd = split_command[0].lower()
            args = split_command[1:]
        except ValueError:
            # Fallback for malformed commands
            parts = command.split()
            cmd = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
        
        # Special handling for mysql -e "query" to correctly parse the query
        if cmd == "mysql" and len(args) >= 2 and args[0] == "-e":
            # Reconstruct the query string, preserving quotes
            query_parts = args[1:]
            # If the original command was `mysql -e \"...\"`, shlex might have removed the outer quotes.
            # We need to reconstruct the query as a single string.
            if len(query_parts) > 0:
                # If there are arguments after -e, assume they form the query
                query_str_reconstructed = " ".join(query_parts)
                # Remove surrounding quotes if they exist
                if query_str_reconstructed.startswith('\"') and query_str_reconstructed.endswith('\"'):
                    query_str_reconstructed = query_str_reconstructed[1:-1]
                elif query_str_reconstructed.startswith('\'') and query_str_reconstructed.endswith('\''):
                    query_str_reconstructed = query_str_reconstructed[1:-1]
                args = ["-e", query_str_reconstructed]
            else:
                args = ["-e", ""] # Empty query if only -e is provided

        print(f"DEBUG: Processing command: '{command}', parsed cmd: '{cmd}', args: {args}")

        # Handle database commands if currently in a simulated database session
        # Prioritize interactive MySQL commands if session is active
        if hasattr(self, '_in_mysql_session') and self._in_mysql_session:
            # Even for non-SQL commands like 'clear' when in mysql session
            if cmd == "clear" or cmd in ["exit", "\q", "help", "\h", "show", "select"]:
                response = self._handle_interactive_mysql_command(command, client_ip)
                if response is not None:
                    return response
        
        # Handle different command types
        if cmd in self.static_responses:
            response = self.static_responses[cmd]
            logger.log_command(client_ip, command, "static")
            return response
        
        elif cmd == "ls":
            return self._handle_ls(args, client_ip)
        
        elif cmd == "cat":
            return self._handle_cat(args, client_ip)
        
        elif cmd == "cd":
            return self._handle_cd(args, client_ip)
        
        elif cmd == "ps":
            return self._handle_ps(args, client_ip)
        
        elif cmd == "netstat":
            return self._handle_netstat(args, client_ip)
        
        elif cmd == "tail":
            return self._handle_tail(args, client_ip)
        
        elif cmd == "grep":
            return self._handle_grep(args, client_ip)
        
        elif cmd == "mysql" or cmd == "psql":
            return self._handle_database(cmd, args, client_ip)
        
        elif cmd == "nmap":
            return self._handle_nmap(args, client_ip)
        
        elif cmd in ["history", "env", "printenv"]:
            return self._handle_info_commands(cmd, args, client_ip)
        
        else:
            # Use AI to generate response for unknown commands
            return self._handle_unknown_command(command, client_ip)
    
    def _handle_ls(self, args: List[str], client_ip: str) -> str:
        """Handle ls command."""
        path_arg = args[0] if args and not args[0].startswith('-') else self.filesystem.current_dir
        
        target = self.filesystem.navigate_to_path(path_arg)
        
        if target is None:
            return f"ls: cannot access '{path_arg}': No such file or directory"
        if target["type"] != "directory":
            # If it's a file, just return its name, similar to how `ls` on a file works
            return path_arg.split('/')[-1]

        items = self.filesystem.list_directory(path_arg)
        
        # Get AI-generated directory content if the directory is logically empty or to augment
        from src.ai_engine.text_generator import get_text_generator
        text_gen = get_text_generator()
        
        ai_generated_output = text_gen.generate_command_output(f"ls {path_arg}")
        
        # Split output by lines and then by whitespace to get potential names
        raw_ai_names = []
        for line in ai_generated_output.split('\n'):
            line_stripped = line.strip()
            if line_stripped:
                # Attempt to extract names, splitting by multiple spaces or tabs
                potential_names = re.split(r'\s\s+', line_stripped)
                raw_ai_names.extend([name.strip() for name in potential_names if name.strip()])

        # Filter out names that are already in static items and filter unrealistic names
        static_names = {item["name"] for item in items}
        
        # Define stricter patterns for unrealistic names (e.g., Windows paths, special characters, too short/long)
        unrealistic_patterns = [
            r'^\s*$', # Empty strings
            r'.*[\\/].*\.exe$', # Windows-like paths or paths with .exe
            r'.*\\.*', # Contains backslash anywhere (indicative of Windows paths)
            r'^C:|^D:|^E:', # Windows drive letters
            r'.*\.dll$', r'.*\.sys$', r'.*\.bat$', r'.*\.cmd$', # Other Windows files
            r'^[\W_]{1,2}$', # Very short non-alphanumeric (e.g. $, @, #)
            r'^[\d]{1,2}$', # Very short numbers
            r'.*[\<\>\:"\|\?\*\\]+.*', # Invalid characters in Unix filenames
            r'^admin@honeypot-server:~$', # SSH prompt-like strings
            r'^(?:http|https|ftp)://.*', # URLs
            r'.*/bin/python.*', # Common Python executable paths
            r'.*/usr/libexec/.*', # Common system binary paths
            r'.*Windows\.NET.*', # Windows specific
            r'^\\[\\w]+\\[\\w]+$', # Windows share paths \\SERVER\\SHARE
        ]

        # Define a whitelist of acceptable AI-generated names if the AI produces something unexpected.
        # These are simple, common Unix-like names not typically in our static filesystem.
        whitelist_names = [
            "README.md", "install.sh", "server.log", "database.bak", "index.html",
            "app.py", "main.go", "script.js", "nginx.conf", "apache.conf",
            "users.db", "access.log", "error.log", "docs", "backup",
            "temp", "old", "new", "test", "install", "setup", "update", "share",
            "local", "repo", "src", "include", "lib", "man", "games", "bin",
            "boot", "dev", "media", "mnt", "opt", "proc", "run", "sbin", "srv", "sys"
        ]
        
        # Filter raw_ai_names more aggressively based on whitelist and existing static names
        cleaned_ai_names = []
        for name in raw_ai_names:
            name_cleaned = name.strip()
            # Only consider names that are alphanumeric or contain common file extensions, and are in whitelist
            if name_cleaned and name_cleaned not in static_names and \
               (re.match(r"^[a-zA-Z0-9_.-]+$", name_cleaned) and name_cleaned in whitelist_names): # Strict whitelist check
                cleaned_ai_names.append(name_cleaned)
        
        unique_ai_names = list(set(cleaned_ai_names)) # Remove duplicates
        unique_ai_names = [name for name in unique_ai_names if name] # Ensure no empty strings
        
        # Limit the number of AI-generated items to prevent overwhelming output
        unique_ai_names = unique_ai_names[:random.randint(2, 5)] # Generate 2 to 5 additional items

        # Add AI-generated items to the list
        for name in unique_ai_names:
            # Assign a random type (file or directory) and size/permissions
            item_type = "directory" if random.random() > 0.7 else "file"
            size = random.randint(100, 50000) if item_type == "file" else 4096
            permissions = "drwxr-xr-x" if item_type == "directory" else "-rw-r--r--"
            items.append({"name": name, "type": item_type, "size": size, "permissions": permissions})

        # Sort items for consistent output
        items.sort(key=lambda x: (x["type"], x["name"].lower())) # Sort by type then name
        
        if not items:
            return ""
        
        # Format output
        if "-l" in args:
            # Long format
            output_lines = []
            
            # Dynamically determine max widths for columns for perfect alignment
            # Ensure at least a base width for common fields
            max_perm = max(len(item["permissions"]) for item in items) if items else 10
            max_links = max(len(str(1)) for _ in items) if items else 1 # Always 1 for simplicity
            max_owner = max(len("admin") for _ in items) if items else 5 # Assuming owner is 'admin'
            max_group = max(len("admin") for _ in items) if items else 5 # Assuming group is 'admin'
            max_size = max(len(str(item["size"])) for item in items) if items else 10
            max_date = max(len(time.strftime("%b %d %H:%M")) for _ in items) if items else 12 # Common date format length
            # Use max(max_name, 1) to avoid error on empty list for max_name
            max_name = max(len(item["name"]) for item in items) if items else 1
            

            # Add a header for ls -l
            header_line = (
                f"Permissions{' ' * (max_perm - len('Permissions'))} "
                f"Links{' ' * (max_links - len('Links') + 1)} "
                f"Owner{' ' * (max_owner - len('Owner') + 1)} "
                f"Group{' ' * (max_group - len('Group') + 1)} "
                f"Size{' ' * (max_size - len('Size') + 1)} "
                f"Date{' ' * (max_date - len('Date') + 1)} "
                f"Name"
            )
            # We skip the header for now to mimic simple ls -l output, which doesn't usually print this header
            # output_lines.append(header_line)

            for item in items:
                size_str = str(item["size"]).rjust(max_size)
                date_str = time.strftime("%b %d %H:%M")
                
                # Format each column with appropriate padding
                output_lines.append(
                    f"{item['permissions'].ljust(max_perm)} "
                    f"1 ".ljust(max_links + 1) + # Links
                    f"admin ".ljust(max_owner + 1) + # Owner
                    f"admin ".ljust(max_group + 1) + # Group
                    f"{size_str} "
                    f"{date_str} "
                    f"{item['name']}"
                )
            get_logger().log_command(client_ip, f"ls {' '.join(args)}", "ai_generated")
            return "\n".join(output_lines)
        else:
            # Simple format (multiple columns)
            terminal_width = 80 # Assuming a standard terminal width for output formatting
            names = [item["name"] for item in items]
            if not names:
                return ""

            # Ensure names are not too long for formatting, truncate if necessary
            max_display_name_len = 25 # Max characters for a name in simple ls
            display_names = [
                (name[:max_display_name_len-3] + "...") if len(name) > max_display_name_len else name
                for name in names
            ]

            max_name_len = max(len(name) for name in display_names)
            
            # Adjust max_name_len if it's too small for multi-column layout
            min_col_width = max_name_len + 2 # Name + 2 spaces padding
            num_columns = max(1, terminal_width // min_col_width)
            
            # Recalculate max_name_len to fit available columns if needed (distribute width evenly)
            actual_max_name_len = (terminal_width // num_columns) - 2
            if actual_max_name_len < max_name_len: # If columns are too narrow
                max_name_len = actual_max_name_len
            
            formatted_output_lines = []
            for i in range(0, len(display_names), num_columns):
                row_items = display_names[i:i + num_columns]
                formatted_row = "".join(name.ljust(min_col_width) for name in row_items)
                formatted_output_lines.append(formatted_row.rstrip()) # Remove trailing spaces if row is short

            get_logger().log_command(client_ip, f"ls {' '.join(args)}", "ai_generated")
            return "\n".join(formatted_output_lines)
    
    def _handle_cat(self, args: List[str], client_ip: str) -> str:
        """Handle cat command with AI-generated content."""
        if not args:
            return "cat: missing file operand"
        
        filename = args[0]
        
        # Check if file exists in our filesystem
        current_path_obj = Path(self.filesystem.current_dir)
        target_path_obj = (current_path_obj / filename).resolve()
        
        # Convert to simplified Unix-like string for filesystem lookup
        normalized_filename = str(target_path_obj).replace('\\', '/')
        if not normalized_filename.startswith('/'):
            normalized_filename = '/' + normalized_filename

        if not self.filesystem.file_exists(normalized_filename) and not self.filesystem.file_exists(filename):
            print(f"DEBUG: File '{filename}' not found in filesystem.")
            return f"cat: {filename}: No such file or directory"
        
        # Log file access
        from src.utils.logger import get_logger
        logger = get_logger()
        logger.log_file_access(client_ip, filename, "read")
        
        # Check if we have cached content
        if filename in self.filesystem.file_contents:
            print(f"DEBUG: Returning cached content for '{filename}'")
            return self.filesystem.file_contents[filename]
        
        # Generate content using AI
        try:
            content = ""
            response_type = "gpt2_generated" # Default

            # Determine if it's a CSV or SQL database file and use the appropriate generator
            if filename.lower().endswith('.csv') or filename.lower().endswith('.sql'):
                print(f"DEBUG: _handle_cat: Attempting CTGAN generation for tabular file '{filename}'")
                from src.ai_engine.tabular_generator import get_tabular_generator
                tabular_gen = get_tabular_generator()
                
                # Map specific filenames to schema names
                if filename.lower() == 'customer_data.csv' or filename.lower() == 'customers.sql':
                    actual_table_name = 'customers'
                elif filename.lower() == 'transactions.sql':
                    actual_table_name = 'transactions'
                elif filename.lower() == 'users.sql':
                    actual_table_name = 'users'
                elif filename.lower() == 'logs.sql':
                    actual_table_name = 'logs'
                else:
                    actual_table_name = filename.replace('.csv', '').replace('.sql', '').replace('.db', '')

                # For demo, generate a fixed number of rows
                generated_tabular_content = tabular_gen.generate_csv_file(actual_table_name, num_rows=10)
                
                if "Error: Could not generate synthetic data" in generated_tabular_content:
                    print(f"DEBUG: CTGAN generation failed for '{filename}', falling back to text generator for error message.")
                    from src.ai_engine.text_generator import get_text_generator
                    text_gen = get_text_generator()
                    content = text_gen.generate_file_content(filename) # Generate text based error/generic content
                    response_type = "gpt2_fallback_for_ctgan_error"
                else:
                    content = generated_tabular_content
                    response_type = "ctgan_generated"
                    print(f"DEBUG: Successfully generated tabular content for '{filename}'. Length: {len(content)}")

            else: # Text-based files
                print(f"DEBUG: _handle_cat: Attempting GPT-2 generation for text file '{filename}'")
                from src.ai_engine.text_generator import get_text_generator
                text_gen = get_text_generator()
                content = text_gen.generate_file_content(filename)
                response_type = "gpt2_generated"
            
            # Cache the content
            self.filesystem.file_contents[filename] = content
            
            logger.log_command(client_ip, f"cat {filename}", response_type)
            return content
            
        except Exception as e:
            print(f"Error generating content for {filename}: {e}")
            logger.log_command(client_ip, f"cat {filename}", "error")
            return f"Error reading file: {filename}"
    
    def _handle_cd(self, args: List[str], client_ip: str) -> str:
        """Handle cd command."""
        if not args:
            target = self.environment_vars["HOME"]
        else:
            target = args[0]
        
        # Resolve '..' and '.' in path
        if target == "..":
            # Go up one level, ensure we don't go above root
            new_path_parts = self.filesystem.current_dir.split('/')
            if len(new_path_parts) > 2: # More than just '/' and an empty string
                target = '/'.join(new_path_parts[:-1])
                if not target: # If it becomes empty, go to root
                    target = "/"
            else: # Already at root or /home/admin and trying to go up
                target = "/" 
        elif target == ".":
            target = self.filesystem.current_dir
        else:
            # Handle relative path
            if not target.startswith('/'):
                target = os.path.join(self.filesystem.current_dir, target).replace('\\', '/')
        
        # Simplified cd handling
        # Ensure target is a valid directory
        path_info = self.filesystem.navigate_to_path(target)
        if path_info is not None and path_info.get("type") == "directory":
                self.filesystem.current_dir = target
                self.environment_vars["PWD"] = target
                get_logger().log_command(client_ip, f"cd {args[0] if args else ''}", "processed")
                return ""
        else:
            get_logger().log_command(client_ip, f"cd {args[0] if args else ''}", "error")
            return f"cd: {args[0] if args else target}: No such file or directory"
    
    def _handle_ps(self, args: List[str], client_ip: str) -> str:
        """Handle ps command with AI-generated process list."""
        try:
            from src.ai_engine.text_generator import get_text_generator
            text_gen = get_text_generator()
            
            # Generate realistic process list
            output = text_gen.generate_command_output("ps aux")
            
            # Add header if not present
            if "PID" not in output and "USER" not in output: # More robust check
                header = "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"
                output = header + "\n" + output
            
            get_logger().log_command(client_ip, "ps " + " ".join(args), "ai_generated")
            return output
            
        except Exception as e:
            print(f"Error generating ps output: {e}")
            return self._fallback_ps_output()
    
    def _handle_netstat(self, args: List[str], client_ip: str) -> str:
        """Handle netstat command."""
        try:
            from src.ai_engine.text_generator import get_text_generator
            text_gen = get_text_generator()
            
            output = text_gen.generate_command_output("netstat -an")
            
            # Ensure proper header
            if "Proto" not in output and "Local Address" not in output: # More robust check
                header = "Proto Recv-Q Send-Q Local Address           Foreign Address         State"
                output = header + "\n" + output
            
            get_logger().log_command(client_ip, "netstat " + " ".join(args), "ai_generated")
            return output
            
        except Exception as e:
            print(f"Error generating netstat output: {e}")
            return self._fallback_netstat_output()
    
    def _handle_tail(self, args: List[str], client_ip: str) -> str:
        """Handle tail command for log files."""
        if not args:
            return "tail: missing file operand"
        
        filename = args[0]
        
        # Check if it's a log file
        if "log" in filename.lower():
            try:
                from src.ai_engine.log_generator import get_log_generator
                log_gen = get_log_generator()
                
                if "auth" in filename.lower(): # Case-insensitive check
                    content = log_gen.generate_auth_log(10)
                elif "sys" in filename.lower(): # Case-insensitive check
                    content = log_gen.generate_syslog(10)
                else:
                    # Generate generic log entries
                    entries = log_gen.generate_log_sequence(num_hours=1)
                    content = "\n".join([
                        f"{entry['timestamp']} {entry['level']} {entry['service']}: {entry['message']}"
                        for entry in entries[-10:]  # Last 10 entries
                    ])
                
                get_logger().log_file_access(client_ip, filename, "tail")
                get_logger().log_command(client_ip, f"tail {filename}", "ai_generated")
                return content
                
            except Exception as e:
                print(f"Error generating log content for '{filename}': {e}")
                # Fallback to text generator for logs if log_generator fails
                from src.ai_engine.text_generator import get_text_generator
                text_gen = get_text_generator()
                content = text_gen.generate_file_content(filename)
                get_logger().log_command(client_ip, f"tail {filename}", "gpt2_fallback_for_log_error")
                return content
        
        # Fall back to regular file handling
        return self._handle_cat([filename], client_ip)
    
    def _handle_grep(self, args: List[str], client_ip: str) -> str:
        """Handle grep command."""
        if len(args) < 2:
            return "grep: missing pattern or file"
        
        pattern = args[0]
        filename = args[1]
        
        # Get file content first
        content = self._handle_cat([filename], client_ip)
        
        if content.startswith("cat:"): # Propagate error from cat
            return content.replace("cat:", "grep:")
        
        # Simple pattern matching
        matching_lines = [
            line for line in content.split("\n")
            if pattern.lower() in line.lower()
        ]
        
        get_logger().log_command(client_ip, f"grep {pattern} {filename}", "processed")
        return "\n".join(matching_lines[:20])  # Limit output
    
    # Track if we are in an interactive MySQL session
    _in_mysql_session: bool = False
    _mysql_prompt: str = "mysql> "

    def _handle_interactive_mysql_command(self, command: str, client_ip: str) -> Optional[str]:
        """Handles commands when inside the interactive MySQL monitor."""
        command_lower = command.lower().strip()
        logger = get_logger()

        if command_lower == "exit" or command_lower == "\\q":
            self._in_mysql_session = False
            logger.log_command(client_ip, command, "mysql_exit")
            return "Goodbye!"
        elif command_lower == "clear": # Handle 'clear' command in MySQL session
            logger.log_command(client_ip, command, "mysql_clear")
            return ""
        elif command_lower.startswith("help") or command_lower.startswith("\\h"):
            logger.log_command(client_ip, command, "mysql_help")
            return "MySQL monitor help: Commands end with ; or '\\g'. \nAvailable commands: show databases, show tables, select, help, exit."
        elif command_lower.startswith("show databases"):
            # Remove trailing semicolon for comparison if present
            if command_lower.endswith(';'):
                command_lower = command_lower[:-1]
            if command_lower == "show databases": # Strict comparison after cleaning
                logger.log_command(client_ip, command, "mysql_show_databases")
                # Format as proper MySQL table
                databases = ["information_schema", "mysql", "performance_schema", "sys", "customers_db", "transactions_db"]
                
                # Calculate width for database names
                max_width = max(len(db) for db in databases)
                db_width = max(max_width, len("Database")) + 2
                
                # Create formatted table
                separator = "+" + "-" * db_width + "+"
                header = f"| {'Database'.ljust(db_width-2)} |"
                
                db_rows = []
                for db in databases:
                    db_rows.append(f"| {db.ljust(db_width-2)} |")
                
                response = "\n".join([separator, header, separator] + db_rows + [separator])
                response += f"\n\n{len(databases)} rows in set (0.00 sec)"
                return response
            else:
                return f"ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '{command.split()[0]}...' at line 1"
        elif command_lower.startswith("show tables"):
            # Remove trailing semicolon for comparison if present
            if command_lower.endswith(';'):
                command_lower = command_lower[:-1]
            if command_lower == "show tables": # Strict comparison after cleaning
                logger.log_command(client_ip, command, "mysql_show_tables")
                from src.ai_engine.tabular_generator import get_tabular_generator
                tabular_gen = get_tabular_generator()
                tables = tabular_gen.list_available_tables()
                
                # Format as proper MySQL table
                if tables:
                    # Calculate width for table names
                    max_width = max(len(table) for table in tables)
                    table_width = max(max_width, len("Tables_in_database")) + 2
                    
                    # Create formatted table
                    separator = "+" + "-" * table_width + "+"
                    header = f"| {'Tables_in_database'.ljust(table_width-2)} |"
                    
                    table_rows = []
                    for table in tables:
                        table_rows.append(f"| {table.ljust(table_width-2)} |")
                    
                    response = "\n".join([separator, header, separator] + table_rows + [separator])
                    response += f"\n\n{len(tables)} row{'s' if len(tables) != 1 else ''} in set (0.00 sec)"
                    return response
                else:
                    return "Empty set (0.00 sec)"
            else:
                return f"ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '{command.split()[0]}...' at line 1"
        elif command_lower.startswith("select"):
            # This is a simplified regex to extract table name for demonstration
            match = re.search(r"from\s+(\w+)", command_lower + ";") # Add ; to ensure regex match
            table_name = match.group(1) if match else "customers" # Default to customers

            logger.log_command(client_ip, command, f"mysql_select_{table_name}")
            from src.ai_engine.tabular_generator import get_tabular_generator
            tabular_gen = get_tabular_generator()
            synthetic_data = tabular_gen.generate_synthetic_data(table_name, 5)
            if synthetic_data is not None:
                return self._format_mysql_output(synthetic_data)
            else:
                return f"ERROR: Could not generate synthetic data for table '{table_name}'."
        else:
            logger.log_command(client_ip, command, "mysql_unknown")
            return f"ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '{command.split()[0]}...' at line 1" # Mimic MySQL error
    
    def _handle_database(self, db_type: str, args: List[str], client_ip: str) -> str:
        """Handle database commands with synthetic data generation."""
        try:
            from src.ai_engine.tabular_generator import get_tabular_generator
            tabular_gen = get_tabular_generator()
            
            # Simulate database connection and query
            if db_type == "mysql":
                # If no -e flag or entering interactive monitor
                if not args or args[0] != "-e":
                    self._in_mysql_session = True
                    return "Welcome to the MySQL monitor. Type 'help;' or '\\h' for help.\n" + self._mysql_prompt
                
                # Extract query from -e flag. shlex.split should have already handled quotes.
                query = args[1].strip() # args[1] should now directly be the query string
                print(f"DEBUG: MySQL -e query extracted: '{query}'")

                # Clean the query for robust comparison
                cleaned_query_lower = query.lower().strip().rstrip(';')

                # Handle specific MySQL commands directly
                if "show databases" in cleaned_query_lower:
                    # Format as proper MySQL table
                    databases = ["information_schema", "mysql", "performance_schema", "sys", "customers_db", "transactions_db"]
                    
                    # Calculate width for database names
                    max_width = max(len(db) for db in databases)
                    db_width = max(max_width, len("Database")) + 2
                    
                    # Create formatted table
                    separator = "+" + "-" * db_width + "+"
                    header = f"| {'Database'.ljust(db_width-2)} |"
                    
                    db_rows = []
                    for db in databases:
                        db_rows.append(f"| {db.ljust(db_width-2)} |")
                    
                    response = "\n".join([separator, header, separator] + db_rows + [separator])
                    response += f"\n\n{len(databases)} rows in set (0.00 sec)"
                elif "help" in cleaned_query_lower:
                    response = "MySQL monitor help: Commands end with ; or '\\g'. \nAvailable commands: show databases, show tables, select, help." # Simple help text
                elif "show tables" in cleaned_query_lower:
                    tables = tabular_gen.list_available_tables()
                    
                    # Format as proper MySQL table
                    if tables:
                        # Calculate width for table names
                        max_width = max(len(table) for table in tables)
                        table_width = max(max_width, len("Tables_in_database")) + 2
                        
                        # Create formatted table
                        separator = "+" + "-" * table_width + "+"
                        header = f"| {'Tables_in_database'.ljust(table_width-2)} |"
                        
                        table_rows = []
                        for table in tables:
                            table_rows.append(f"| {table.ljust(table_width-2)} |")
                        
                        response = "\n".join([separator, header, separator] + table_rows + [separator])
                        response += f"\n\n{len(tables)} row{'s' if len(tables) != 1 else ''} in set (0.00 sec)"
                    else:
                        response = "Empty set (0.00 sec)"
                elif "select" in cleaned_query_lower:
                    # Determine which table to query
                    match = re.search(r"from\s+(\w+)", query.lower())
                    table_name = match.group(1) if match else "customers" # Default
                    
                    # Generate synthetic data
                    synthetic_data = tabular_gen.generate_synthetic_data(table_name, 5)
                    if synthetic_data is not None:
                        response = self._format_mysql_output(synthetic_data)
                    else:
                        response = f"ERROR: Could not generate synthetic data for table '{table_name}'."
                else:
                    response = f"Query '{query}' executed successfully on {db_type} database (simulated)." # Generic success for other queries
                
            get_logger().log_command(client_ip, f"{db_type} " + " ".join(args), "ai_generated")
            return response
            
        except Exception as e:
            print(f"Error handling database command: {e}")
            get_logger().log_command(client_ip, f"{db_type} " + " ".join(args), "error")
            return f"Error connecting to database: {e}"
    
    def _format_mysql_output(self, df) -> str:
        """Format DataFrame as MySQL table output with proper ASCII table formatting."""
        if df.empty:
            return "Empty set (0.00 sec)"
        
        # Limit to first 5 rows for display
        display_df = df.head(5)
        
        # Convert all values to strings and handle None/NaN values
        df_str = display_df.astype(str).replace('nan', 'NULL').replace('None', 'NULL')
        
        # Calculate column widths (minimum 4 characters, add padding)
        col_widths = []
        for col in df_str.columns:
            # Get max width between column name and data values
            col_name_width = len(str(col))
            data_width = df_str[col].apply(lambda x: len(str(x))).max() if not df_str[col].empty else 0
            col_widths.append(max(col_name_width, data_width, 4) + 2)  # Add 2 for padding
        
        # Create separators
        top_separator = "+" + "+".join("-" * width for width in col_widths) + "+"
        
        # Create header row
        header_cells = []
        for col, width in zip(df_str.columns, col_widths):
            cell_content = f" {str(col).ljust(width-2)} "
            header_cells.append(cell_content)
        header = "|" + "|".join(header_cells) + "|"
        
        # Create data rows
        data_rows = []
        for _, row in df_str.iterrows():
            row_cells = []
            for val, width in zip(row, col_widths):
                cell_content = f" {str(val).ljust(width-2)} "
                row_cells.append(cell_content)
            data_rows.append("|" + "|".join(row_cells) + "|")
        
        # Assemble the complete table
        table_parts = [top_separator, header, top_separator] + data_rows + [top_separator]
        
        # Add row count information
        row_count = len(display_df)
        total_rows = len(df)
        if total_rows > 5:
            footer = f"{row_count} rows in set (showing first {row_count} of {total_rows}) (0.00 sec)"
        else:
            footer = f"{row_count} row{'s' if row_count != 1 else ''} in set (0.00 sec)"
        
        return "\n".join(table_parts) + "\n\n" + footer
    
    def _handle_nmap(self, args: List[str], client_ip: str) -> str:
        """Handle nmap command with fake scan results."""
        target = args[0] if args else "localhost"
        
        fake_ports = [
            "22/tcp   open  ssh",
            "80/tcp   open  http", 
            "443/tcp  open  https",
            "3306/tcp open  mysql",
            "5432/tcp open  postgresql",
            "6379/tcp open  redis",
            "8080/tcp open  http-proxy"
        ]
        
        # Randomize which ports appear open
        open_ports = random.sample(fake_ports, random.randint(3, 6))
        
        output = f"""Starting Nmap 7.80 ( https://nmap.org ) at {time.strftime('%Y-%m-%d %H:%M %Z')}
Nmap scan report for {target}
Host is up (0.0034s latency).
Not shown: 993 closed ports
PORT     STATE SERVICE
"""
        output += "\n".join(open_ports)
        output += f"\n\nNmap done: 1 IP address (1 host up) scanned in {random.uniform(1.2, 3.8):.2f} seconds"
        
        get_logger().log_command(client_ip, f"nmap {' '.join(args)}", "simulated")
        return output
    
    def _handle_info_commands(self, cmd: str, args: List[str], client_ip: str) -> str:
        """Handle information commands like history, env."""
        if cmd == "history":
            return "\n".join([f"  {i+1}  {cmd}" for i, cmd in enumerate(self.command_history[-20:])])
        
        elif cmd in ["env", "printenv"]:
            return "\n".join([f"{key}={value}" for key, value in self.environment_vars.items()])
        
        return f"{cmd}: command handled"
    
    def _handle_unknown_command(self, command: str, client_ip: str) -> str:
        """Handle unknown commands by returning a 'command not found' message.
        AI generation is not used for truly unknown commands to avoid nonsensical output.
        """
        get_logger().log_command(client_ip, command, "unknown_command")
        return f"{command.split()[0]}: command not found" # Only return the first word of the command
    
    def _fallback_ps_output(self) -> str:
        """Fallback ps output when AI generation fails."""
        return """USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root         1  0.0  0.4  77616  8784 ?        Ss   Sep01   0:02 /sbin/init
root         2  0.0  0.0      0     0 ?        S    Sep01   0:00 [kthreadd]
admin     1234  0.1  2.1 123456 43210 pts/0    S    15:30   0:05 python3 honeypot.py
admin     5678  0.0  1.2  98765 24680 pts/1    R    15:42   0:00 ps aux"""
    
    def _fallback_netstat_output(self) -> str:
        """Fallback netstat output when AI generation fails."""
        return """Proto Recv-Q Send-Q Local Address           Foreign Address         State      
tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN     
tcp        0      0 0.0.0.0:80              0.0.0.0:*               LISTEN     
tcp        0      0 127.0.0.1:3306          0.0.0.0:*               LISTEN     
tcp        0      0 192.168.1.100:22        192.168.1.50:54321      ESTABLISHED"""

# Global instance
command_handler = None

def get_command_handler() -> CommandHandler:
    """Get global command handler instance."""
    global command_handler
    if command_handler is None:
        command_handler = CommandHandler()
    return command_handler
