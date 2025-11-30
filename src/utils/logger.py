"""Enhanced logging utilities for honeypot activity tracking."""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import threading

class HoneypotLogger:
    """Centralized logging system for honeypot activities."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Thread-safe activity storage
        self._activities = []
        self._lock = threading.Lock()
        
        # Setup different loggers
        self.setup_loggers()
        
    def setup_loggers(self):
        """Setup specialized loggers for different components."""
        
        # Main honeypot logger
        self.honeypot_logger = logging.getLogger("honeypot")
        self.honeypot_logger.setLevel(logging.INFO)
        
        # AI activity logger
        self.ai_logger = logging.getLogger("ai_engine")
        self.ai_logger.setLevel(logging.INFO)
        
        # Attack logger
        self.attack_logger = logging.getLogger("attacks")
        self.attack_logger.setLevel(logging.INFO)
        
        # Setup handlers
        for logger_name, logger in [
            ("honeypot", self.honeypot_logger),
            ("ai_engine", self.ai_logger),
            ("attacks", self.attack_logger)
        ]:
            handler = logging.FileHandler(self.log_dir / f"{logger_name}.log")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def log_connection(self, client_ip: str, connection_type: str = "SSH"):
        """Log incoming connection attempts."""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": "connection",
            "client_ip": client_ip,
            "connection_type": connection_type,
            "status": "connected"
        }
        
        with self._lock:
            self._activities.append(activity)
            
        self.attack_logger.info(f"New {connection_type} connection from {client_ip}")
        
    def log_command(self, client_ip: str, command: str, response_type: str = "static"):
        """Log executed commands and response generation."""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": "command",
            "client_ip": client_ip,
            "command": command,
            "response_type": response_type
        }
        
        with self._lock:
            self._activities.append(activity)
            
        self.honeypot_logger.info(f"Command '{command}' executed by {client_ip}")
        
    def log_ai_generation(self, generator_type: str, content_type: str, 
                         prompt: str, generation_time: float):
        """Log AI content generation activities."""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": "ai_generation",
            "generator": generator_type,
            "content_type": content_type,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "generation_time": round(generation_time, 3)
        }
        
        with self._lock:
            self._activities.append(activity)
            
        self.ai_logger.info(
            f"{generator_type} generated {content_type} in {generation_time:.3f}s"
        )
    
    def log_file_access(self, client_ip: str, file_path: str, access_type: str):
        """Log file access attempts."""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": "file_access",
            "client_ip": client_ip,
            "file_path": file_path,
            "access_type": access_type
        }
        
        with self._lock:
            self._activities.append(activity)
            
        self.honeypot_logger.info(f"File {access_type}: {file_path} by {client_ip}")
    
    def get_recent_activities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent activities for dashboard display."""
        with self._lock:
            return self._activities[-limit:] if self._activities else []
    
    def get_activity_stats(self) -> Dict[str, Any]:
        """Generate activity statistics."""
        with self._lock:
            activities = self._activities.copy()
        
        if not activities:
            return {"total": 0, "by_type": {}, "unique_ips": 0}
        
        stats = {
            "total": len(activities),
            "by_type": {},
            "unique_ips": len(set(
                act.get("client_ip", "unknown") for act in activities 
                if "client_ip" in act
            ))
        }
        
        for activity in activities:
            activity_type = activity.get("type", "unknown")
            stats["by_type"][activity_type] = stats["by_type"].get(activity_type, 0) + 1
        
        return stats

# Global logger instance
honeypot_logger = None

def get_logger() -> HoneypotLogger:
    """Get global logger instance."""
    global honeypot_logger
    if honeypot_logger is None:
        from config import LOGS_DIR
        honeypot_logger = HoneypotLogger(LOGS_DIR)
    return honeypot_logger
