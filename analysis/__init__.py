"""
Computational Model for Behavioral Analysis
Transforms raw trace data into high-level structured context
"""

__version__ = '1.0.0'

from .database_connector import DatabaseConnector
from .vectorizer import EventVectorizer
from .sequence_processor import SequenceProcessor

__all__ = ['DatabaseConnector', 'EventVectorizer', 'SequenceProcessor']

