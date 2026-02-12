"""
Database Connector for Computational Model
Supports both SQLite and PostgreSQL databases
"""

import sqlite3
import json
import os
from typing import Any
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import config


class DatabaseConnector:
    """Connects to companion service database (SQLite or PostgreSQL)"""
    
    def __init__(self):
        self.db_type = config.DATABASE_TYPE
        self.engine = None
        self.sqlite_conn = None
        
        if self.db_type == 'postgres':
            if not config.DATABASE_URL:
                raise ValueError("DATABASE_URL required for PostgreSQL")
            self.engine = create_engine(config.DATABASE_URL)
        else:
            # SQLite
            db_path = config.DATABASE_PATH
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database not found: {db_path}")
            self.sqlite_conn = sqlite3.connect(db_path)
            self.sqlite_conn.row_factory = sqlite3.Row
    
    def execute_query(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute query and return results as list of dictionaries"""
        if self.db_type == 'postgres':
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result]
        else:
            # SQLite
            cursor = self.sqlite_conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_events(self, workspace_path: Optional[str] = None, 
                   limit: Optional[int] = None,
                   order_by: str = 'timestamp ASC') -> List[Dict]:
        """Get events from database"""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if workspace_path:
            query += " AND workspace_path = ?"
            params.append(workspace_path)
        
        query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query, tuple(params))
    
    def get_prompts(self, workspace_path: Optional[str] = None,
                   limit: Optional[int] = None,
                   order_by: str = 'timestamp ASC') -> List[Dict]:
        """Get prompts from database"""
        query = "SELECT * FROM prompts WHERE 1=1"
        params = []
        
        if workspace_path:
            query += " AND workspace_path = ?"
            params.append(workspace_path)
        
        query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query, tuple(params))
    
    def get_entries(self, workspace_path: Optional[str] = None,
                   limit: Optional[int] = None,
                   order_by: str = 'timestamp ASC') -> List[Dict]:
        """Get entries (file changes) from database"""
        query = "SELECT * FROM entries WHERE 1=1"
        params = []
        
        if workspace_path:
            query += " AND workspace_path = ?"
            params.append(workspace_path)
        
        query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query, tuple(params))
    
    def get_events_with_prompts(self, workspace_path: Optional[str] = None,
                               limit: Optional[int] = None) -> List[Dict]:
        """Get events joined with their associated prompts"""
        if self.db_type == 'postgres':
            query = """
                SELECT 
                    e.*,
                    p.text as prompt_text,
                    p.context_files_json,
                    p.context_file_count,
                    p.context_window_size
                FROM events e
                LEFT JOIN prompts p ON e.prompt_id = p.id
                WHERE 1=1
            """
        else:
            query = """
                SELECT 
                    e.*,
                    p.text as prompt_text,
                    p.context_files_json,
                    p.context_file_count,
                    p.context_window_size
                FROM events e
                LEFT JOIN prompts p ON CAST(e.prompt_id AS INTEGER) = p.id
                WHERE 1=1
            """
        
        params = []
        if workspace_path:
            query += " AND e.workspace_path = ?"
            params.append(workspace_path)
        
        query += " ORDER BY e.timestamp ASC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query, tuple(params))
    
    def get_entries_for_prompt(self, prompt_id: int, 
                               time_window_seconds: int = 300) -> List[Dict]:
        """Get entries (file changes) associated with a prompt within time window"""
        # Get prompt timestamp
        prompt_query = "SELECT timestamp FROM prompts WHERE id = ?"
        prompt_result = self.execute_query(prompt_query, (prompt_id,))
        
        if not prompt_result:
            return []
        
        prompt_time = prompt_result[0]['timestamp']
        
        # Calculate time window
        if self.db_type == 'postgres':
            query = """
                SELECT DISTINCT file_path
                FROM entries
                WHERE (prompt_id = :prompt_id 
                    OR timestamp BETWEEN 
                        (SELECT timestamp FROM prompts WHERE id = :prompt_id) - INTERVAL ':window seconds'
                        AND (SELECT timestamp FROM prompts WHERE id = :prompt_id) + INTERVAL ':window seconds')
                    AND file_path IS NOT NULL
                    AND (before_code != after_code OR before_code IS NULL)
            """
            return self.execute_query(query, {
                'prompt_id': prompt_id,
                'window': time_window_seconds
            })
        else:
            # SQLite - use datetime arithmetic
            from datetime import datetime, timedelta
            try:
                prompt_dt = datetime.fromisoformat(prompt_time.replace('Z', '+00:00'))
            except:
                # Fallback parsing
                prompt_dt = datetime.strptime(prompt_time.split('.')[0], '%Y-%m-%dT%H:%M:%S')
            
            window_start = (prompt_dt - timedelta(seconds=time_window_seconds)).isoformat()
            window_end = (prompt_dt + timedelta(seconds=time_window_seconds)).isoformat()
            
            query = """
                SELECT DISTINCT file_path
                FROM entries
                WHERE (prompt_id = ? OR (timestamp >= ? AND timestamp <= ?))
                    AND file_path IS NOT NULL
                    AND (before_code != after_code OR before_code IS NULL)
            """
            return self.execute_query(query, (prompt_id, window_start, window_end))
    
    def close(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.engine:
            self.engine.dispose()

