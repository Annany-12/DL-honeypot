"""TimeGAN-based log sequence generation for realistic system logs."""

import numpy as np
import pandas as pd
import time
import random
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json

# Try to import TimeGAN, fall back to LSTM if not available
try:
    from ydata_synthetic.synthesizers import ModelParameters
    from ydata_synthetic.synthesizers.timeseries import TimeGAN
    TIMEGAN_AVAILABLE = True
except ImportError:
    print("TimeGAN not available, using LSTM-based fallback")
    TIMEGAN_AVAILABLE = False
    import torch
    import torch.nn as nn

class LogSequenceGenerator:
    """TimeGAN/LSTM-powered log sequence generator for realistic system logs."""
    
    def __init__(self):
        """Initialize the log sequence generator."""
        self.models = {}
        self.log_patterns = self._define_log_patterns()
        self.sequence_length = 24  # Hours in a day
        self.n_features = 5  # log_level, service_id, user_id, hour, event_type
        
        # Setup device for GPU support
        self.device = self._setup_device()
        
        print(f"Log Sequence Generator initialized on device: {self.device}")
    
    def _setup_device(self):
        """Setup the appropriate device (GPU/CPU) for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"GPU detected for log generation: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("No GPU detected for log generation, using CPU")
        
        return device
    
    def _define_log_patterns(self) -> Dict[str, Any]:
        """Define common log patterns and templates."""
        return {
            "levels": ["INFO", "WARN", "ERROR", "DEBUG"],
            "services": ["web_server", "database", "auth_service", "api_gateway", "cache", "worker"],
            "event_types": ["login", "logout", "request", "error", "start", "stop", "update"],
            "templates": {
                "INFO": [
                    "User {user_id} logged in successfully from {ip}",
                    "Request processed for endpoint /{endpoint}",
                    "Service {service} started successfully",
                    "Database connection established",
                    "Cache updated for key {key}"
                ],
                "WARN": [
                    "High CPU usage detected: {cpu}%",
                    "Database connection pool nearly full",
                    "Slow query detected: {query_time}ms",
                    "Memory usage above threshold: {memory}%"
                ],
                "ERROR": [
                    "Authentication failed for user {user_id}",
                    "Database connection lost",
                    "Service {service} failed to start",
                    "API request timeout for {endpoint}",
                    "File not found: {filename}"
                ],
                "DEBUG": [
                    "Processing request with ID {request_id}",
                    "Cache miss for key {key}",
                    "Query execution time: {query_time}ms",
                    "Memory allocation: {memory}MB"
                ]
            }
        }
    
    def _generate_training_sequences(self, num_sequences: int = 100) -> np.ndarray:
        """Generate training sequences for TimeGAN/LSTM."""
        sequences = []
        
        for _ in range(num_sequences):
            sequence = []
            
            # Generate a day's worth of log data (24 hours)
            for hour in range(24):
                # Simulate realistic patterns
                if 2 <= hour <= 6:  # Night time - less activity
                    activity_level = 0.2
                elif 9 <= hour <= 17:  # Business hours - high activity
                    activity_level = 1.0
                else:  # Evening/morning - moderate activity
                    activity_level = 0.6
                
                # Generate log entry features
                log_level = random.choices(
                    [0, 1, 2, 3],  # INFO, WARN, ERROR, DEBUG
                    weights=[0.6, 0.2, 0.1, 0.1],
                    k=1
                )[0]
                
                service_id = random.randint(0, len(self.log_patterns["services"]) - 1)
                user_id = random.randint(1, 100) if random.random() > 0.3 else 0
                event_type = random.randint(0, len(self.log_patterns["event_types"]) - 1)
                
                # Scale activity by time of day
                activity_factor = activity_level * random.uniform(0.5, 1.5)
                
                sequence.append([
                    log_level,
                    service_id,
                    user_id / 100.0,  # Normalize
                    hour / 24.0,  # Normalize
                    event_type,
                    activity_factor
                ])
            
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def train_timegan_model(self, model_name: str = "system_logs", 
                           num_sequences: int = None, epochs: int = None) -> bool:
        """Train TimeGAN model on log sequences."""
        # Load configuration
        from config import (TIMEGAN_SEQ_LEN, TIMEGAN_HIDDEN_DIM, TIMEGAN_GAMMA, 
                           TIMEGAN_BATCH_SIZE, TIMEGAN_LEARNING_RATE)
        
        num_sequences = num_sequences or 200
        epochs = epochs or 50
        
        if not TIMEGAN_AVAILABLE:
            return self._train_lstm_model(model_name, num_sequences, epochs)
        
        start_time = time.time()
        
        try:
            print(f"Generating training sequences for {model_name}...")
            print(f"Configuration - Sequences: {num_sequences}, Epochs: {epochs}")
            training_data = self._generate_training_sequences(num_sequences)
            
            print(f"Training TimeGAN model with shape: {training_data.shape}")
            
            # Configure TimeGAN parameters
            gan_args = ModelParameters(
                batch_size=min(TIMEGAN_BATCH_SIZE, num_sequences // 4),
                lr=TIMEGAN_LEARNING_RATE,
                noise_dim=32,
                layers_dim=128
            )
            
            # Initialize TimeGAN
            timegan = TimeGAN(
                model_parameters=gan_args,
                hidden_dim=TIMEGAN_HIDDEN_DIM,
                seq_len=TIMEGAN_SEQ_LEN,
                n_seq=training_data.shape[2],
                gamma=TIMEGAN_GAMMA
            )
            
            # Train the model
            timegan.train(training_data, train_steps=epochs)
            
            # Store the trained model
            self.models[model_name] = timegan
            
            training_time = time.time() - start_time
            
            from src.utils.logger import get_logger
            get_logger().log_ai_generation(
                "TimeGAN", f"{model_name}_model_training", 
                f"{num_sequences} sequences", training_time
            )
            
            print(f"TimeGAN model for {model_name} trained successfully in {training_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"Error training TimeGAN model for {model_name}: {e}")
            return False
    
    def _train_lstm_model(self, model_name: str, num_sequences: int, epochs: int) -> bool:
        """Fallback LSTM training when TimeGAN is not available."""
        print(f"Training LSTM fallback model for {model_name}...")
        
        try:
            # Generate training data
            training_data = self._generate_training_sequences(num_sequences)
            
            # Create a simple LSTM-based model using PyTorch
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size):
                    super(SimpleLSTM, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                    
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out)
                    return out
            
            # Load configuration
            from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_LEARNING_RATE, LSTM_EPOCHS
            
            # Initialize model
            input_size = training_data.shape[2]
            hidden_size = LSTM_HIDDEN_SIZE
            num_layers = LSTM_NUM_LAYERS
            output_size = input_size
            
            model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)
            model = model.to(self.device)  # Move model to appropriate device
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
            
            # Convert training data to tensor and move to device
            train_tensor = torch.FloatTensor(training_data).to(self.device)
            
            # Simple training loop
            model.train()
            max_epochs = min(epochs, LSTM_EPOCHS)
            for epoch in range(max_epochs):
                optimizer.zero_grad()
                outputs = model(train_tensor)
                loss = criterion(outputs, train_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"LSTM Epoch {epoch}/{max_epochs}, Loss: {loss.item():.4f}")
            
            # Store the trained model
            self.models[model_name] = {
                'model': model,
                'model_type': 'lstm_fallback',
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers
            }
            
            print(f"LSTM fallback model for {model_name} trained successfully")
            return True
            
        except Exception as e:
            print(f"Error training LSTM fallback model: {e}")
            # Create a simple pattern-based fallback
            self.models[model_name] = {
                'model_type': 'pattern_based',
                'patterns': self.log_patterns
            }
            return True
    
    def generate_log_sequence(self, model_name: str = "system_logs", 
                             num_hours: int = 24) -> List[Dict[str, Any]]:
        """Generate a sequence of realistic log entries."""
        start_time = time.time()
        
        # Train model if not already trained
        if model_name not in self.models:
            print(f"Training model {model_name}...")
            if not self.train_timegan_model(model_name):
                return self._generate_fallback_sequence(num_hours)
        
        try:
            model_info = self.models[model_name]
            
            if TIMEGAN_AVAILABLE and isinstance(model_info, object) and hasattr(model_info, 'sample'):
                # Use TimeGAN for generation
                synthetic_sequences = model_info.sample(1)
                sequence_data = synthetic_sequences[0]
            elif isinstance(model_info, dict) and model_info.get('model_type') == 'lstm_fallback':
                # Use LSTM fallback for generation
                sequence_data = self._generate_lstm_sequence(model_info, num_hours)
            elif isinstance(model_info, dict) and model_info.get('model_type') == 'pattern_based':
                # Use pattern-based fallback
                sequence_data = self._generate_fallback_sequence_data(num_hours)
            else:
                # Use fallback method
                sequence_data = self._generate_fallback_sequence_data(num_hours)
            
            # Convert numerical sequence to readable log entries
            log_entries = self._convert_sequence_to_logs(sequence_data, num_hours)
            
            generation_time = time.time() - start_time
            
            from src.utils.logger import get_logger
            get_logger().log_ai_generation(
                "TimeGAN/LSTM", "log_sequence", 
                f"{num_hours} hours", generation_time
            )
            
            return log_entries
            
        except Exception as e:
            print(f"Error generating log sequence: {e}")
            return self._generate_fallback_sequence(num_hours)
    
    def _convert_sequence_to_logs(self, sequence_data: np.ndarray, num_hours: int) -> List[Dict[str, Any]]:
        """Convert numerical sequence data to readable log entries."""
        logs = []
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        for i in range(min(num_hours, len(sequence_data))):
            if len(sequence_data[i]) >= 5:
                log_level_idx = int(sequence_data[i][0]) % len(self.log_patterns["levels"])
                service_idx = int(sequence_data[i][1]) % len(self.log_patterns["services"])
                user_id = int(sequence_data[i][2] * 100) if sequence_data[i][2] > 0.01 else None
                event_type_idx = int(sequence_data[i][4]) % len(self.log_patterns["event_types"])
                
                level = self.log_patterns["levels"][log_level_idx]
                service = self.log_patterns["services"][service_idx]
                event_type = self.log_patterns["event_types"][event_type_idx]
                
                # Generate timestamp
                log_time = base_time + timedelta(hours=i, minutes=random.randint(0, 59))
                
                # Generate message from template
                message = self._generate_log_message(level, service, user_id, event_type)
                
                log_entry = {
                    "timestamp": log_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "level": level,
                    "service": service,
                    "message": message,
                    "user_id": user_id,
                    "event_type": event_type
                }
                
                logs.append(log_entry)
        
        return logs
    
    def _generate_log_message(self, level: str, service: str, user_id: Optional[int], 
                            event_type: str) -> str:
        """Generate a realistic log message based on parameters."""
        templates = self.log_patterns["templates"].get(level, ["Generic {level} message"])
        template = random.choice(templates)
        
        # Fill in template variables
        replacements = {
            "user_id": user_id or random.randint(1, 100),
            "service": service,
            "ip": f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
            "endpoint": random.choice(["api/users", "api/data", "login", "dashboard", "reports"]),
            "cpu": random.randint(60, 95),
            "memory": random.randint(60, 90),
            "query_time": random.randint(100, 5000),
            "request_id": f"req_{random.randint(10000, 99999)}",
            "key": f"cache_key_{random.randint(1000, 9999)}",
            "filename": f"data_{random.randint(1, 100)}.csv"
        }
        
        try:
            return template.format(**replacements)
        except KeyError:
            return template
    
    def _generate_lstm_sequence(self, model_info: dict, num_hours: int) -> np.ndarray:
        """Generate sequence using trained LSTM model."""
        try:
            model = model_info['model']
            model.eval()
            
            # Create initial sequence
            initial_sequence = self._generate_fallback_sequence_data(min(24, num_hours))
            input_tensor = torch.FloatTensor(initial_sequence).unsqueeze(0).to(model['model'].device if hasattr(model['model'], 'device') else torch.device('cpu'))
            
            with torch.no_grad():
                generated = model(input_tensor)
                sequence_data = generated.squeeze(0).numpy()
            
            # Extend or truncate to desired length
            if len(sequence_data) < num_hours:
                # Repeat pattern if needed
                repeats = (num_hours // len(sequence_data)) + 1
                sequence_data = np.tile(sequence_data, (repeats, 1))
            
            return sequence_data[:num_hours]
            
        except Exception as e:
            print(f"Error generating LSTM sequence: {e}")
            return self._generate_fallback_sequence_data(num_hours)
    
    def _generate_fallback_sequence_data(self, num_hours: int) -> np.ndarray:
        """Generate fallback sequence data when TimeGAN is not available."""
        sequence = []
        
        for hour in range(num_hours):
            # Realistic activity patterns
            if 2 <= hour <= 6:  # Night
                activity = 0.2
            elif 9 <= hour <= 17:  # Business hours
                activity = 1.0
            else:  # Evening/morning
                activity = 0.6
            
            log_level = random.choices([0, 1, 2, 3], weights=[0.6, 0.2, 0.1, 0.1])[0]
            service_id = random.randint(0, len(self.log_patterns["services"]) - 1)
            user_id = random.randint(1, 100) / 100.0 if random.random() > 0.3 else 0
            event_type = random.randint(0, len(self.log_patterns["event_types"]) - 1)
            
            sequence.append([log_level, service_id, user_id, hour / 24.0, event_type, activity])
        
        return np.array(sequence)
    
    def _generate_fallback_sequence(self, num_hours: int) -> List[Dict[str, Any]]:
        """Generate fallback log sequence when AI generation fails."""
        logs = []
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        for i in range(num_hours):
            level = random.choice(self.log_patterns["levels"])
            service = random.choice(self.log_patterns["services"])
            user_id = random.randint(1, 100) if random.random() > 0.3 else None
            event_type = random.choice(self.log_patterns["event_types"])
            
            log_time = base_time + timedelta(hours=i, minutes=random.randint(0, 59))
            message = self._generate_log_message(level, service, user_id, event_type)
            
            logs.append({
                "timestamp": log_time.strftime("%Y-%m-%d %H:%M:%S"),
                "level": level,
                "service": service,
                "message": message,
                "user_id": user_id,
                "event_type": event_type
            })
        
        return logs
    
    def generate_auth_log(self, num_entries: int = 20) -> str:
        """Generate realistic authentication log entries."""
        logs = []
        
        for _ in range(num_entries):
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 7),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            ip = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
            user = f"user{random.randint(1, 50)}"
            
            if random.random() > 0.8:  # 20% failed attempts
                status = "FAILED"
                message = f"authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost={ip} user={user}"
            else:
                status = "SUCCESS"
                message = f"Accepted password for {user} from {ip} port {random.randint(30000, 65000)} ssh2"
            
            log_entry = f"{timestamp.strftime('%b %d %H:%M:%S')} honeypot sshd[{random.randint(1000, 9999)}]: {message}"
            logs.append(log_entry)
        
        return '\n'.join(sorted(logs))
    
    def generate_syslog(self, num_entries: int = 30) -> str:
        """Generate realistic system log entries."""
        logs = []
        
        services = ["kernel", "systemd", "NetworkManager", "cron", "apache2", "mysql"]
        
        for _ in range(num_entries):
            timestamp = datetime.now() - timedelta(
                days=random.randint(0, 7),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            service = random.choice(services)
            pid = random.randint(100, 9999)
            
            messages = {
                "kernel": [
                    "Out of memory: Kill process {pid} (python) score {score} or sacrifice child",
                    "CPU{cpu}: Core temperature above threshold, cpu clock throttled"
                ],
                "systemd": [
                    "Started {service}.service",
                    "Stopping {service}.service"
                ],
                "NetworkManager": [
                    "device (eth0): state change: activated -> deactivating",
                    "dhcp4 (eth0): option requested_routers => {ip}"
                ],
                "cron": [
                    "({user}) CMD ({cmd})",
                    "pam_unix(cron:session): session opened for user {user} by (uid=0)"
                ]
            }
            
            message_templates = messages.get(service, ["Generic service message"])
            message = random.choice(message_templates).format(
                pid=pid,
                score=random.randint(1, 1000),
                cpu=random.randint(0, 7),
                service=random.choice(["backup", "cleanup", "update"]),
                ip=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                user=f"user{random.randint(1, 20)}",
                cmd=random.choice(["/usr/bin/updatedb", "/home/backup.sh", "python3 /opt/monitor.py"])
            )
            
            log_entry = f"{timestamp.strftime('%b %d %H:%M:%S')} honeypot {service}[{pid}]: {message}"
            logs.append(log_entry)
        
        return '\n'.join(sorted(logs))

# Global instance
log_generator = None

def get_log_generator() -> LogSequenceGenerator:
    """Get global log generator instance."""
    global log_generator
    if log_generator is None:
        log_generator = LogSequenceGenerator()
    return log_generator
