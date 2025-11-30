"""Main entry point for the AI-Driven Honeypot Simulator."""

import uvicorn
import threading
import time
import asyncio
from pathlib import Path
import sys # Added for sys.argv

from config import SSH_PORT, API_PORT, DASHBOARD_PORT, BASE_DIR
from src.honeypot.ssh_server import SSHHoneypot
from src.ai_engine.text_generator import get_text_generator
from src.ai_engine.tabular_generator import get_tabular_generator
from src.ai_engine.log_generator import get_log_generator
from src.utils.logger import get_logger
from src.utils.data_loader import get_data_loader

# Initialize global components (these are singletons)
logger = get_logger() # Initialize logger first
data_loader = get_data_loader()
text_generator = get_text_generator()
tabular_generator = get_tabular_generator()
log_generator = get_log_generator()

# Global variable to hold the SSH Honeypot instance
ssh_honeypot_instance: SSHHoneypot = None

async def start_fastapi_server():
    """Starts the FastAPI server for potential API interactions."""
    logger.honeypot_logger.info(f"Starting FastAPI server on port {API_PORT}")
    print(f"FastAPI server (simulated) running on http://127.0.0.1:{API_PORT}")

def start_honeypot_components():
    """Starts the core honeypot components (SSH server and simulated FastAPI)."""
    global ssh_honeypot_instance
    logger.honeypot_logger.info("Starting AI-driven Honeypot Simulator components...")

    # 1. Start SSH Honeypot Server in a separate thread
    if ssh_honeypot_instance is None:
        ssh_honeypot_instance = SSHHoneypot(port=SSH_PORT)
        ssh_thread = threading.Thread(target=ssh_honeypot_instance.start, daemon=True)
        ssh_thread.start()
        logger.honeypot_logger.info(f"SSH Honeypot listening on port {SSH_PORT}")
        print(f"SSH Honeypot listening on port {SSH_PORT}")
    else:
        logger.honeypot_logger.info("SSH Honeypot already running.")
        print("SSH Honeypot already running.")

    # 2. Start FastAPI server (simulated for now)
    print(f"FastAPI server (placeholder) would run on http://127.0.0.1:{API_PORT}")
    logger.honeypot_logger.info(f"FastAPI server (placeholder) would run on http://127.0.0.1:{API_PORT}")

def stop_honeypot_components():
    """Stops the core honeypot components."""
    global ssh_honeypot_instance
    if ssh_honeypot_instance:
        ssh_honeypot_instance.stop()
        ssh_honeypot_instance = None
        logger.honeypot_logger.info("Honeypot Simulator components stopped.")
        print("Honeypot Simulator components stopped.")

def run_honeypot_main():
    """Main function when run directly (not by Streamlit)."""
    start_honeypot_components()
    # For direct execution, keep the main thread alive for daemon threads
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.honeypot_logger.info("Honeypot Simulator shutting down.")
        stop_honeypot_components()


if __name__ == "__main__":
    # Ensure the 'models' and 'data' directories exist as per config.py
    (BASE_DIR / "models").mkdir(exist_ok=True)
    (BASE_DIR / "data").mkdir(exist_ok=True)
    (BASE_DIR / "logs").mkdir(exist_ok=True)

    # Determine if running via Streamlit
    is_streamlit = False
    for arg in sys.argv:
        if "streamlit" in arg:
            is_streamlit = True
            break

    if is_streamlit:
        # When run by streamlit, only prepare for components to be started by dashboard
        print("Running via Streamlit. Honeypot components will be managed by the dashboard.")
    else:
        # When run directly, start all components (for local testing)
        print("Running main.py directly. Starting all components (SSH, FastAPI placeholder).")
        run_honeypot_main()
