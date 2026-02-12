"""
Configuration for Computational Model
Reads from environment variables or .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_TYPE = os.getenv('DATABASE_TYPE', 'sqlite')  # 'sqlite' or 'postgres'
DATABASE_URL = os.getenv('DATABASE_URL', None)

# SQLite path (relative to companion service root)
if DATABASE_TYPE == 'sqlite':
    # Default to companion service data directory
    COMPANION_ROOT = Path(__file__).parent.parent
    DEFAULT_DB_PATH = COMPANION_ROOT / 'data' / 'companion.db'
    DATABASE_PATH = os.getenv('DATABASE_PATH', str(DEFAULT_DB_PATH))
else:
    DATABASE_PATH = None

# Embedding service configuration
EMBEDDING_SERVICE = os.getenv('EMBEDDING_SERVICE', 'openrouter')  # 'openrouter', 'huggingface', 'local'
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
HF_TOKEN = os.getenv('HF_TOKEN', os.getenv('HUGGINGFACE_API_KEY', ''))
EMBEDDING_MODEL = os.getenv('CLIO_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
HF_ENDPOINT = os.getenv('HF_EMBEDDING_ENDPOINT', None)

# Clustering configuration
CLUSTERING_METHOD = os.getenv('CLUSTERING_METHOD', 'dtw')  # 'dtw', 'kmeans', 'hierarchical'
MIN_CLUSTER_SIZE = int(os.getenv('MIN_CLUSTER_SIZE', '3'))
MAX_CLUSTERS = int(os.getenv('MAX_CLUSTERS', '20'))

# Context Precision configuration
CP_TIME_WINDOW_SECONDS = int(os.getenv('CP_TIME_WINDOW_SECONDS', '300'))  # 5 minutes
MIN_CP_THRESHOLD = float(os.getenv('MIN_CP_THRESHOLD', '0.5'))

# Output configuration
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', Path(__file__).parent / 'output'))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

