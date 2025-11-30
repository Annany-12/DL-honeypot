"""SSH server implementation for the honeypot."""

import asyncio
import socket
import threading
import paramiko
from io import StringIO
import time
import traceback
from typing import Dict, Optional
from src.utils.logger import get_logger

class HoneypotSSHServer(paramiko.ServerInterface):
    """SSH server interface for honeypot."""
    
    def __init__(self, client_ip: str):
        self.client_ip = client_ip
        self.authenticated = False
        
    def check_auth_password(self, username: str, password: str) -> int:
        """Check password authentication."""
        from src.utils.logger import get_logger
        
        # Log authentication attempt
        get_logger().honeypot_logger.info(f"SSH auth attempt: {username}:{password} from {self.client_ip}")
        
        # Accept common weak passwords for demonstration
        weak_passwords = ["admin", "password", "123456", "admin123", "root", ""]
        
        if password in weak_passwords:
            self.authenticated = True
            get_logger().honeypot_logger.info(f"SSH auth SUCCESS: {username} from {self.client_ip}")
            return paramiko.AUTH_SUCCESSFUL
        else:
            get_logger().honeypot_logger.info(f"SSH auth FAILED: {username} from {self.client_ip}")
            return paramiko.AUTH_FAILED
    
    def get_allowed_auths(self, username: str) -> str:
        """Return allowed authentication methods."""
        return "password"
    
    def check_channel_request(self, kind: str, chanid: int) -> int:
        """Handle channel requests."""
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
    
    def check_channel_pty_request(self, channel, term, width, height, pixelwidth, pixelheight, modes):
        # Always accept PTY requests so clients can get an interactive shell
        return True

    def check_channel_shell_request(self, channel):
        # Always accept shell requests (the honeypot will provide a fake shell)
        return True



class SSHSession:
    """Handle individual SSH session."""
    
    def __init__(self, client_socket: socket.socket, client_addr: tuple):
        self.client_socket = client_socket
        self.client_addr = client_addr
        self.client_ip = client_addr[0]
        
    def handle_session(self):
        """Handle the SSH session."""
        try:
            from src.utils.logger import get_logger
            get_logger().log_connection(self.client_ip, "SSH")
            
            # Create SSH transport
            transport = paramiko.Transport(self.client_socket)
            
            # Load server key (you'll need to generate this)
            try:
                host_key = paramiko.RSAKey.generate(2048)
                transport.add_server_key(host_key)
            except Exception as e:
                print(f"Failed to generate host key: {e}")
                return
            
            # Create server interface
            server = HoneypotSSHServer(self.client_ip)
            
            try:
                transport.start_server(server=server)
            except paramiko.SSHException as e:
                print(f"SSH negotiation failed: {e}")
                return
            
            # Wait for authentication
            chan = transport.accept(20)
            if chan is None:
                print("No channel request")
                return

            if not server.authenticated:
                print("Client not authenticated")
                chan.close()
                return

            shell_thread = threading.Thread(target=self.handle_shell_session, args=(chan,), daemon=True)
            shell_thread.start()
            shell_thread.join() # Wait for the shell session to complete

        except Exception as e:
            print(f"SSH session error: {e}")
            traceback.print_exc()
        finally:
            try:
                self.client_socket.close()
            except:
                pass
    
    def handle_shell_session(self, chan: paramiko.Channel):
        """Handle interactive shell session."""
        from src.honeypot.command_handler import get_command_handler

        command_handler = get_command_handler()
        
        try:
            # Send welcome message
            chan.send("Welcome to Ubuntu 20.04.3 LTS (GNU/Linux 5.4.0-42-generic x86_64)\r\n")
            chan.send(" * Documentation:  https://help.ubuntu.com\r\n")
            chan.send(" * Management:     https://landscape.canonical.com\r\n")
            chan.send(" * Support:        https://ubuntu.com/advantage\r\n\r\n")
            chan.send("admin@honeypot-server:~$ ")
            
            command_buffer = ""
            
            while True:
                try:
                    # Wait for data
                    if chan.recv_ready():
                        data = chan.recv(1024).decode('utf-8', errors='ignore')
                        
                        if not data:
                            break
                        
                        # Handle special characters
                        for char in data:
                            if char == '\r' or char == '\n':
                                # Execute command
                                if command_buffer.strip():
                                    if command_buffer.strip().lower() in ['exit', 'quit', 'logout']:
                                        chan.send("\r\nGoodbye!\r\n")
                                        return
                                    
                                    # Process command
                                    result = command_handler.handle_command(command_buffer.strip(), self.client_ip)
                                    
                                    if result:
                                        chan.send(f"\r\n{result}\r\n")
                                    else:
                                        chan.send("\r\n")
                                
                                # Reset command buffer and show prompt
                                command_buffer = ""
                                chan.send("admin@honeypot-server:~$ ")
                                
                            elif char == '\x7f' or char == '\b':  # Backspace
                                if command_buffer:
                                    command_buffer = command_buffer[:-1]
                                    chan.send("\b \b")  # Erase character
                                    
                            elif char == '\x03':  # Ctrl+C
                                command_buffer = ""
                                chan.send("\r\n^C\r\nadmin@honeypot-server:~$ ")
                                
                            elif char.isprintable():
                                command_buffer += char
                                chan.send(char)  # Echo character
                    
                    # Check if connection is still alive
                    if chan.closed:
                        break
                        
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Shell session error: {e}")
                    break
                    
        except Exception as e:
            print(f"Shell handling error: {e}")
        finally:
            try:
                chan.close()
            except:
                pass

class SSHHoneypot:
    """Main SSH honeypot server."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 2222):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None
        
    def start(self):
        """Start the SSH honeypot server."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            
            self.running = True
            
            print(f"SSH Honeypot started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, client_addr = self.server_socket.accept()
                    
                    print(f"New SSH connection from {client_addr[0]}:{client_addr[1]}")
                    
                    # Handle session in a separate thread
                    session = SSHSession(client_socket, client_addr)
                    session_thread = threading.Thread(
                        target=session.handle_session,
                        daemon=True
                    )
                    session_thread.start()
                    
                except Exception as e:
                    if self.running:
                        print(f"SSH server error: {e}")
                    
        except Exception as e:
            print(f"Failed to start SSH server: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the SSH honeypot server."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("SSH Honeypot stopped")

def start_ssh_honeypot(host: str = "0.0.0.0", port: int = 2222):
    """Start SSH honeypot in a separate thread."""
    honeypot = SSHHoneypot(host, port)
    
    server_thread = threading.Thread(target=honeypot.start, daemon=True)
    server_thread.start()
    
    return honeypot
