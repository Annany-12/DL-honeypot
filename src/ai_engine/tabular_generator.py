"""CTGAN-based tabular data generation for fake databases."""

import pandas as pd
import numpy as np
from ctgan import CTGAN
import time
import random
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import torch # Import torch for tensor operations
import os

class TabularDataGenerator:
    """CTGAN-powered synthetic tabular data generator."""
    
    def __init__(self):
        """Initialize the tabular data generator."""
        self.models = {}  # Store trained models
        self.schemas = self._define_common_schemas()
        
        # Configure device for GPU support
        self.device = self._setup_device()
        print(f"CTGAN Tabular Generator initialized on device: {self.device}")
    
    def _setup_device(self):
        """Setup the appropriate device (GPU/CPU) for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            print("No GPU detected, using CPU")
        
        return device
    
    def _define_common_schemas(self) -> Dict[str, Dict]:
        """Define common database schemas for synthetic generation."""
        return {
            "customers": {
                "columns": ["customer_id", "name", "email", "phone", "age", "city", "country", "registration_date"],
                "types": ["int", "str", "str", "str", "int", "str", "str", "str"],
                "discrete": ["name", "email", "phone", "city", "country", "registration_date"],
                "sample_data": self._generate_sample_customers()
            },
            "transactions": {
                "columns": ["transaction_id", "customer_id", "amount", "currency", "status", "timestamp", "merchant"],
                "types": ["int", "int", "float", "str", "str", "str", "str"],
                "discrete": ["currency", "status", "timestamp", "merchant"],
                "sample_data": self._generate_sample_transactions()
            },
            "users": {
                "columns": ["user_id", "username", "email", "role", "last_login", "status", "department"],
                "types": ["int", "str", "str", "str", "str", "str", "str"],
                "discrete": ["username", "email", "role", "last_login", "status", "department"],
                "sample_data": self._generate_sample_users()
            },
            "logs": {
                "columns": ["log_id", "timestamp", "level", "service", "message", "ip_address", "user_id"],
                "types": ["int", "str", "str", "str", "str", "str", "int"],
                "discrete": ["timestamp", "level", "service", "message", "ip_address"],
                "sample_data": self._generate_sample_logs()
            }
        }
    
    def _generate_sample_customers(self) -> pd.DataFrame:
        """Generate sample customer data for training."""
        np.random.seed(42)
        n_samples = 1000
        
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa", "James", "Maria"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
        countries = ["USA", "Canada", "UK", "Germany", "France", "Australia", "Japan", "Brazil", "India", "Mexico"]
        
        data = {
            "customer_id": range(1, n_samples + 1),
            "name": [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_samples)],
            "email": [f"user{i}@example.com" for i in range(n_samples)],
            "phone": [f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}" for _ in range(n_samples)],
            "age": np.random.randint(18, 80, n_samples),
            "city": [random.choice(cities) for _ in range(n_samples)],
            "country": [random.choice(countries) for _ in range(n_samples)],
            "registration_date": [f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}" for _ in range(n_samples)]
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_transactions(self) -> pd.DataFrame:
        """Generate sample transaction data for training."""
        np.random.seed(42)
        n_samples = 2000
        
        currencies = ["USD", "EUR", "GBP", "CAD", "AUD"]
        statuses = ["completed", "pending", "failed", "cancelled"]
        merchants = ["Amazon", "PayPal", "Stripe", "Square", "Walmart", "Target", "Best Buy", "Apple", "Google", "Microsoft"]
        
        data = {
            "transaction_id": range(1, n_samples + 1),
            "customer_id": np.random.randint(1, 1001, n_samples),
            "amount": np.round(np.random.lognormal(3, 1), 2),
            "currency": [random.choice(currencies) for _ in range(n_samples)],
            "status": [random.choice(statuses) for _ in range(n_samples)],
            "timestamp": [f"2024-09-{random.randint(1,30):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}" for _ in range(n_samples)],
            "merchant": [random.choice(merchants) for _ in range(n_samples)]
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_users(self) -> pd.DataFrame:
        """Generate sample user data for training."""
        np.random.seed(42)
        n_samples = 500
        
        roles = ["admin", "user", "moderator", "guest", "analyst", "developer"]
        statuses = ["active", "inactive", "suspended", "pending"]
        departments = ["IT", "Finance", "HR", "Marketing", "Sales", "Operations", "Support"]
        
        data = {
            "user_id": range(1, n_samples + 1),
            "username": [f"user_{i}" for i in range(n_samples)],
            "email": [f"employee{i}@company.com" for i in range(n_samples)],
            "role": [random.choice(roles) for _ in range(n_samples)],
            "last_login": [f"2024-09-{random.randint(1,30):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}" for _ in range(n_samples)],
            "status": [random.choice(statuses) for _ in range(n_samples)],
            "department": [random.choice(departments) for _ in range(n_samples)]
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_logs(self) -> pd.DataFrame:
        """Generate sample log data for training."""
        np.random.seed(42)
        n_samples = 5000
        
        levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        services = ["web_server", "database", "auth_service", "api_gateway", "cache", "queue", "worker"]
        messages = [
            "User login successful",
            "Database connection established",
            "Request processed",
            "Cache miss occurred",
            "Service started",
            "Configuration updated",
            "Backup completed",
            "Error in processing",
            "Authentication failed"
        ]
        
        data = {
            "log_id": range(1, n_samples + 1),
            "timestamp": [f"2024-09-{random.randint(1,30):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}" for _ in range(n_samples)],
            "level": [random.choice(levels) for _ in range(n_samples)],
            "service": [random.choice(services) for _ in range(n_samples)],
            "message": [random.choice(messages) for _ in range(n_samples)],
            "ip_address": [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(n_samples)],
            "user_id": np.random.randint(1, 501, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def train_model(self, table_name: str, epochs: int = None) -> bool:
        """Train CTGAN model on specified table schema."""
        # Load configuration
        from config import (CTGAN_EPOCHS, CTGAN_BATCH_SIZE, CTGAN_GENERATOR_DIM, 
                           CTGAN_DISCRIMINATOR_DIM, CTGAN_LEARNING_RATE)
        
        epochs = epochs or CTGAN_EPOCHS
        start_time = time.time()
        
        if table_name not in self.schemas:
            print(f"Unknown table schema: {table_name}")
            return False
        
        try:
            schema = self.schemas[table_name]
            training_data = schema["sample_data"]
            discrete_columns = schema["discrete"]
            
            print(f"Training CTGAN model for {table_name}...")
            print(f"Training data shape: {training_data.shape}")
            print(f"Configuration - Epochs: {epochs}, Batch size: {CTGAN_BATCH_SIZE}")
            
            # Initialize CTGAN with configurable parameters and GPU support
            batch_size = min(CTGAN_BATCH_SIZE, len(training_data) // 2)
            if self.device.type == 'cuda':
                # Adjust batch size for GPU memory
                available_memory = torch.cuda.get_device_properties(0).total_memory
                if available_memory < 4 * 1024**3:  # Less than 4GB
                    batch_size = min(batch_size, 128)
                print(f"Using GPU with batch size: {batch_size}")
            
            ctgan = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                generator_dim=CTGAN_GENERATOR_DIM,
                discriminator_dim=CTGAN_DISCRIMINATOR_DIM,
                verbose=True,
                cuda=self.device.type == 'cuda'
            )
            
            print(f"Training CTGAN for {epochs} epochs...")
            
            # Train the model properly using the official API
            ctgan.fit(training_data, discrete_columns)
            
            # Store the trained model (losses are handled internally by CTGAN)
            self.models[table_name] = {
                'model': ctgan,
                'training_epochs': epochs,
                'training_samples': len(training_data),
                'discrete_columns': discrete_columns
            }
            
            training_time = time.time() - start_time
            
            from src.utils.logger import get_logger
            get_logger().log_ai_generation(
                "CTGAN", f"{table_name}_model_training", 
                f"{len(training_data)} rows", training_time
            )
            
            print(f"CTGAN model for {table_name} trained successfully in {training_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"Error training CTGAN model for {table_name}: {e}")
            return False
    
    def generate_synthetic_data(self, table_name: str, num_rows: int = 100) -> Optional[pd.DataFrame]:
        """Generate synthetic data for specified table."""
        start_time = time.time()
        
        if table_name not in self.schemas:
            print(f"Unknown table schema: {table_name}")
            return None
        
        # Train model if not already trained
        if table_name not in self.models:
            print(f"Training CTGAN model for {table_name}...")
            if not self.train_model(table_name):
                return None
        
        try:
            model_info = self.models[table_name]
            model = model_info['model'] # Access the actual CTGAN model
            
            print(f"Generating {num_rows} synthetic rows for {table_name}...")
            
            # Generate synthetic data
            synthetic_data = model.sample(num_rows)
            
            generation_time = time.time() - start_time
            
            from src.utils.logger import get_logger
            get_logger().log_ai_generation(
                "CTGAN", f"{table_name}_synthetic_data", 
                f"{num_rows} rows", generation_time
            )
            
            print(f"Generated {num_rows} synthetic rows in {generation_time:.2f}s")
            return synthetic_data
            
        except Exception as e:
            print(f"Error generating synthetic data for {table_name}: {e}")
            return self._generate_fallback_data(table_name, num_rows)
    
    def generate_csv_file(self, table_name: str, num_rows: int = 100, 
                         output_path: Optional[Path] = None) -> str:
        """Generate synthetic data and save as CSV file."""
        synthetic_data = self.generate_synthetic_data(table_name, num_rows)
        
        if synthetic_data is None:
            return "Error: Could not generate synthetic data"
        
        # Convert to CSV string
        csv_content = synthetic_data.to_csv(index=False)
        
        # Optionally save to file
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_data.to_csv(output_path, index=False)
            print(f"Synthetic data saved to {output_path}")
        
        return csv_content
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a specific table schema."""
        if table_name not in self.schemas:
            return {"error": f"Unknown table: {table_name}"}
        
        schema = self.schemas[table_name]
        sample_data = schema["sample_data"]
        
        return {
            "table_name": table_name,
            "columns": schema["columns"],
            "column_types": schema["types"],
            "discrete_columns": schema["discrete"],
            "sample_rows": len(sample_data),
            "trained": table_name in self.models,
            "sample_preview": sample_data.head(3).to_dict('records')
        }
    
    def list_available_tables(self) -> List[str]:
        """List all available table schemas."""
        return list(self.schemas.keys())
    
    def _generate_fallback_data(self, table_name: str, num_rows: int) -> pd.DataFrame:
        """Generate fallback data when CTGAN fails."""
        schema = self.schemas[table_name]
        sample_data = schema["sample_data"]
        
        # Simple sampling with noise for fallback
        if len(sample_data) >= num_rows:
            return sample_data.sample(num_rows).reset_index(drop=True)
        else:
            # Repeat and add noise
            repeated_data = pd.concat([sample_data] * (num_rows // len(sample_data) + 1))
            return repeated_data.head(num_rows).reset_index(drop=True)

# Global instance
tabular_generator = None

def get_tabular_generator() -> TabularDataGenerator:
    """Get global tabular generator instance."""
    global tabular_generator
    if tabular_generator is None:
        tabular_generator = TabularDataGenerator()
    return tabular_generator
