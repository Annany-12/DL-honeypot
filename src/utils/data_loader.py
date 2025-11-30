import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and preprocessing various datasets required for the AI models.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataLoader initialized with data directory: {self.data_dir}")

    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Loads a CSV file from the data directory."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"CSV file not found: {filepath}")
            return None
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV file: {filepath} with {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {e}")
            return None

    def save_csv(self, df: pd.DataFrame, filename: str):
        """Saves a DataFrame to a CSV file in the data directory."""
        filepath = self.data_dir / filename
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved CSV file: {filepath}")
        except Exception as e:
            logger.error(f"Error saving CSV {filepath}: {e}")

    def load_text_file(self, filename: str) -> Optional[str]:
        """Loads content from a text file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"Text file not found: {filepath}")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded text file: {filepath}")
            return content
        except Exception as e:
            logger.error(f"Error loading text file {filepath}: {e}")
            return None

    def save_text_file(self, content: str, filename: str):
        """Saves content to a text file."""
        filepath = self.data_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved text file: {filepath}")
        except Exception as e:
            logger.error(f"Error saving text file {filepath}: {e}")

    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Loads a JSON file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"JSON file not found: {filepath}")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON file: {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON {filepath}: {e}")
            return None

    def save_json(self, data: Dict[str, Any], filename: str):
        """Saves data to a JSON file."""
        filepath = self.data_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Saved JSON file: {filepath}")
        except Exception as e:
            logger.error(f"Error saving JSON {filepath}: {e}")

# Global instance for easy access
data_loader = None

def get_data_loader(data_dir: Optional[Path] = None) -> DataLoader:
    """Get global data loader instance."""
    global data_loader
    if data_loader is None:
        if data_dir is None:
            from config import DATA_DIR
            data_dir = DATA_DIR
        data_loader = DataLoader(data_dir)
    return data_loader

