#!/usr/bin/env python3
"""
Fast N8N Workflow Database
SQLite-based workflow indexer and search engine for instant performance.
"""

import sqlite3
import json
import os
import glob
import datetime
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class WorkflowDatabase:
    """High-performance SQLite database for workflow metadata and search."""
    
    def __init__(self, db_path: str = 'workflows.db'):
        # Use environment variable if no path provided
        self.db_path = os.environ.get('WORKFLOW_DB_PATH', db_path)
        self.workflows_dir = "workflows"  # Directory containing workflow files
        self.conn = None
        self.timeout = 60  # Increased timeout for operations
        self.retry_attempts = 5  # Increased retry attempts
        self._initialize_database()
        self.model = None  # Will be loaded on first use
        self.embedding_model = None
        self.stop_words = set(stopwords.words('english'))
        
    def _initialize_database(self):
        """Initialize the database connection and schema."""
        try:
            self.conn = self.get_connection()
            self.init_database()
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
    
    def init_database(self):
        """Initialize SQLite database with optimized schema and indexes."""
        conn = self.get_connection()
        conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging for performance
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        # Create main workflows table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                workflow_id TEXT,
                active BOOLEAN DEFAULT 0,
                description TEXT,
                trigger_type TEXT,
                complexity TEXT,
                node_count INTEGER DEFAULT 0,
                integrations TEXT,  -- JSON array
                tags TEXT,         -- JSON array
                created_at TEXT,
                updated_at TEXT,
                search_text TEXT,  -- Preprocessed text for semantic search
                embedding BLOB,    -- Vector embedding for semantic search
                file_hash TEXT,
                file_size INTEGER,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add missing columns if they don't exist
        cursor = conn.cursor()
        
        # Check if search_text column exists
        cursor.execute("PRAGMA table_info(workflows)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'search_text' not in columns:
            try:
                conn.execute("ALTER TABLE workflows ADD COLUMN search_text TEXT")
                print("Added search_text column to workflows table")
            except sqlite3.OperationalError as e:
                print(f"Error adding search_text column: {e}")
        
        if 'embedding' not in columns:
            try:
                conn.execute("ALTER TABLE workflows ADD COLUMN embedding BLOB")
                print("Added embedding column to workflows table")
            except sqlite3.OperationalError as e:
                print(f"Error adding embedding column: {e}")
                
        conn.commit()
        
        # Create FTS5 table for full-text search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS workflows_fts USING fts5(
                filename,
                name,
                description,
                integrations,
                tags,
                content=workflows,
                content_rowid=id
            )
        """)
        
        # Create indexes for fast filtering
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trigger_type ON workflows(trigger_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_complexity ON workflows(complexity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_active ON workflows(active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_node_count ON workflows(node_count)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_filename ON workflows(filename)")
        
        # Create triggers to keep FTS table in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS workflows_ai AFTER INSERT ON workflows BEGIN
                INSERT INTO workflows_fts(rowid, filename, name, description, integrations, tags)
                VALUES (new.id, new.filename, new.name, new.description, new.integrations, new.tags);
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS workflows_ad AFTER DELETE ON workflows BEGIN
                INSERT INTO workflows_fts(workflows_fts, rowid, filename, name, description, integrations, tags)
                VALUES ('delete', old.id, old.filename, old.name, old.description, old.integrations, old.tags);
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS workflows_au AFTER UPDATE ON workflows BEGIN
                INSERT INTO workflows_fts(workflows_fts, rowid, filename, name, description, integrations, tags)
                VALUES ('delete', old.id, old.filename, old.name, old.description, old.integrations, old.tags);
                INSERT INTO workflows_fts(rowid, filename, name, description, integrations, tags)
                VALUES (new.id, new.filename, new.name, new.description, new.integrations, new.tags);
            END
        """)
        
        conn.commit()
        conn.close()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection with retry logic."""
        if self.conn is not None:
            try:
                # Test if connection is still valid
                self.conn.execute('SELECT 1')
                return self.conn
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                # Connection is closed or invalid, create a new one
                self.conn = None
        
        for attempt in range(self.retry_attempts):
            try:
                self.conn = sqlite3.connect(
                    self.db_path,
                    timeout=self.timeout,
                    isolation_level='IMMEDIATE',  # Better for concurrent access
                    check_same_thread=False  # Allow multiple threads to access the connection
                )
                self.conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
                self.conn.execute('PRAGMA busy_timeout=60000')  # 60 second timeout
                self.conn.execute('PRAGMA synchronous=NORMAL')  # Better write performance
                self.conn.execute('PRAGMA cache_size=10000')  # Increase cache size
                self.conn.row_factory = sqlite3.Row
                return self.conn
            except sqlite3.OperationalError as e:
                if 'locked' in str(e).lower() and attempt < self.retry_attempts - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"Database locked, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                print(f"Failed to connect to database after {attempt + 1} attempts: {e}")
                raise
        
        raise sqlite3.OperationalError("Failed to establish database connection after multiple retries")
    
    def get_embedding_model(self):
        """Lazy load the embedding model."""
        if self.embedding_model is None:
            # Using a lightweight model for local use
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for semantic search."""
        if not text:
            return ""
            
        try:
            # Ensure NLTK data is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
            # Convert to lowercase
            text = str(text).lower()
            
            # Tokenize and remove stopwords
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Warning: Error in text preprocessing: {str(e)}")
            # Fallback to simple preprocessing
            return str(text).lower().strip()
    
    def _generate_search_text(self, workflow_info: Dict) -> str:
        """Generate search text from workflow info for semantic search."""
        parts = []
        
        # Add name if available
        if 'name' in workflow_info and workflow_info['name']:
            parts.append(workflow_info['name'])
            
        # Add description if available
        if 'description' in workflow_info and workflow_info['description']:
            parts.append(workflow_info['description'])
            
        # Add trigger type if available
        if 'trigger_type' in workflow_info and workflow_info['trigger_type']:
            parts.append(f"Trigger: {workflow_info['trigger_type']}")
            
        # Add integrations if available
        if 'integrations' in workflow_info and workflow_info['integrations']:
            parts.append("Integrations: " + ", ".join(workflow_info['integrations']))
            
        # Add tags if available
        if 'tags' in workflow_info and workflow_info['tags']:
            parts.append("Tags: " + ", ".join(workflow_info['tags']))
            
        # Join all parts with spaces
        return " ".join(parts)
    
    def generate_embedding(self, text: str) -> Optional[bytes]:
        """Generate an embedding for the given text."""
        model = self.get_embedding_model()
        if not model:
            return None
            
        try:
            # Preprocess text
            preprocessed = self.preprocess_text(text)
            if not preprocessed:
                return None
                
            # Generate embedding
            embedding = model.encode(preprocessed, convert_to_numpy=True)
            
            # Convert to bytes for storage
            return embedding.tobytes()
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1: Optional[bytes], vec2: Optional[bytes]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not vec1 or not vec2:
            return 0.0
            
        try:
            vec1 = np.frombuffer(vec1, dtype=np.float32)
            vec2 = np.frombuffer(vec2, dtype=np.float32)
            
            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def search_workflows_semantic(self, query: str, limit: int = 10, threshold: float = 0.3) -> List[Dict]:
        """
        Search workflows using semantic similarity.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching workflows with similarity scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            return []
        
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get all workflows with embeddings
            cursor.execute("""
                SELECT id, filename, name, description, trigger_type, complexity, 
                       node_count, integrations, tags, embedding
                FROM workflows
                WHERE embedding IS NOT NULL
            """)
            
            # Calculate similarity scores
            results = []
            for row in cursor.fetchall():
                similarity = self.cosine_similarity(query_embedding, row['embedding'])
                if similarity >= threshold:
                    result = dict(row)
                    result['similarity'] = similarity
                    results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Convert SQLite Row objects to dicts and clean up
            for result in results:
                if 'embedding' in result:
                    del result['embedding']
                if 'integrations' in result and isinstance(result['integrations'], str):
                    try:
                        result['integrations'] = json.loads(result['integrations'])
                    except (json.JSONDecodeError, TypeError):
                        result['integrations'] = []
                if 'tags' in result and isinstance(result['tags'], str):
                    try:
                        result['tags'] = json.loads(result['tags'])
                    except (json.JSONDecodeError, TypeError):
                        result['tags'] = []
            
            return results[:limit]
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            raise
            
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    print(f"Error closing connection: {e}")
                    pass
    
    def update_workflow_embedding(self, workflow_id: int, workflow_data: Dict):
        """Update the embedding for a workflow."""
        # Create search text from relevant fields
        search_text = ' '.join([
            workflow_data.get('name', ''),
            workflow_data.get('description', ''),
            ' '.join(workflow_data.get('integrations', [])),
            ' '.join(workflow_data.get('tags', [])),
            workflow_data.get('trigger_type', '')
        ])
        
        # Preprocess and generate embedding
        search_text = self.preprocess_text(search_text)
        embedding = self.generate_embedding(search_text)
        
        if not embedding:
            return
            
        # Update database
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE workflows SET search_text = ?, embedding = ? WHERE id = ?",
                (search_text, embedding, workflow_id)
            )
            conn.commit()
        except Exception as e:
            print(f"Error updating workflow embedding: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    print(f"Error closing connection: {e}")
                    
    def _format_workflow_name(self, name: str) -> str:
        """Format workflow name for better readability."""
        if not name:
            return ""
            
        # Split by underscores
        parts = name.split('_')
        
        # Skip the first part if it's just a number
        if len(parts) > 1 and parts[0].isdigit():
            parts = parts[1:]
        
        # Convert parts to title case and join with spaces
        readable_parts = []
        for part in parts:
            # Special handling for common terms
            if part.lower() == 'http':
                readable_parts.append('HTTP')
            elif part.lower() == 'api':
                readable_parts.append('API')
            elif part.lower() == 'webhook':
                readable_parts.append('Webhook')
            elif part.lower() == 'automation':
                readable_parts.append('Automation')
            elif part.lower() == 'automate':
                readable_parts.append('Automate')
            elif part.lower() == 'scheduled':
                readable_parts.append('Scheduled')
            elif part.lower() == 'triggered':
                readable_parts.append('Triggered')
            elif part.lower() == 'manual':
                readable_parts.append('Manual')
            else:
                # Capitalize first letter
                readable_parts.append(part.capitalize())
        
        return ' '.join(readable_parts)
    
    def analyze_workflow_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze a single workflow file and extract metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error reading {file_path}: {str(e)}")
            return None
        
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_hash = self._get_file_hash(file_path)
        
        # Extract basic workflow information
        workflow_info = {
            'filename': filename,
            'name': data.get('name', ''),
            'description': data.get('description', ''),
            'active': data.get('active', False),
            'created_at': data.get('createdAt', ''),
            'updated_at': data.get('updatedAt', '')
        }
        
        # Analyze nodes to get trigger type and integrations
        nodes = data.get('nodes', [])
        trigger_type, integrations = self.analyze_nodes(nodes)
        workflow_info.update({
            'node_count': len(nodes),
            'trigger_type': trigger_type,
            'integrations': list(integrations),
            'tags': data.get('tags', [])
        })
        
        # Determine complexity based on node count and integrations
        if len(nodes) > 20 or len(integrations) > 3:
            workflow_info['complexity'] = 'high'
        elif len(nodes) > 10 or len(integrations) > 1:
            workflow_info['complexity'] = 'medium'
        else:
            workflow_info['complexity'] = 'low'
            
        return workflow_info
        
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def _update_workflow(self, conn, workflow_id: int, workflow_info: dict, file_hash: str, file_size: int) -> None:
        """Update an existing workflow in the database."""
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE workflows 
            SET name = ?, 
                description = ?,
                node_count = ?,
                trigger_type = ?,
                complexity = ?,
                integrations = ?,
                tags = ?,
                file_hash = ?,
                file_size = ?,
                updated_at = CURRENT_TIMESTAMP,
                analyzed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            workflow_info.get('name', ''),
            workflow_info.get('description', ''),
            workflow_info.get('node_count', 0),
            workflow_info.get('trigger_type', 'Manual'),
            workflow_info.get('complexity', 'medium'),
            json.dumps(workflow_info.get('integrations', [])),
            json.dumps(workflow_info.get('tags', [])),
            file_hash,
            file_size,
            workflow_id
        ))
        
    def _insert_workflow(self, conn, workflow_info: dict, file_path: str, file_hash: str, file_size: int) -> None:
        """Insert a new workflow into the database."""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO workflows (
                filename, name, description, node_count, trigger_type,
                complexity, integrations, tags, file_hash, file_size,
                created_at, updated_at, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (
            file_path,
            workflow_info.get('name', os.path.basename(file_path)),
            workflow_info.get('description', ''),
            workflow_info.get('node_count', 0),
            workflow_info.get('trigger_type', 'Manual'),
            workflow_info.get('complexity', 'medium'),
            json.dumps(workflow_info.get('integrations', [])),
            json.dumps(workflow_info.get('tags', [])),
            file_hash,
            file_size
        ))
        
    def analyze_workflow_file(self, file_path: str) -> Optional[Dict]:
        """Analyze a workflow file and return metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            # Basic workflow info
            workflow_info = {
                'filename': os.path.basename(file_path),
                'name': workflow_data.get('name', ''),
                'description': workflow_data.get('description', ''),
                'active': workflow_data.get('active', False),
                'created_at': workflow_data.get('createdAt', ''),
                'updated_at': workflow_data.get('updatedAt', ''),
                'file_hash': self._get_file_hash(file_path),
                'file_size': os.path.getsize(file_path)
            }
            
            # Analyze nodes
            nodes = workflow_data.get('nodes', [])
            workflow_info['node_count'] = len(nodes)
            
            # Extract trigger type (first node type)
            trigger_type = 'Manual'
            if nodes:
                trigger_type = nodes[0].get('type', 'Manual')
            workflow_info['trigger_type'] = trigger_type
            
            # Extract integrations
            integrations = set()
            for node in nodes:
                if 'type' in node:
                    integrations.add(node['type'].split('.')[-1])
            workflow_info['integrations'] = list(integrations)
            
            # Simple complexity estimation
            node_count = workflow_info['node_count']
            if node_count < 5:
                complexity = 'low'
            elif node_count < 15:
                complexity = 'medium'
            else:
                complexity = 'high'
            workflow_info['complexity'] = complexity
            
            # Extract tags if available
            workflow_info['tags'] = workflow_data.get('tags', [])
            
            return workflow_info
            
        except Exception as e:
            print(f"Error analyzing workflow {file_path}: {e}")
            return None
    
    def analyze_nodes(self, nodes: List[Dict]) -> Tuple[str, set]:
        """Analyze nodes to determine trigger type and integrations."""
        trigger_type = 'Manual'
        integrations = set()
        
        # Enhanced service mapping for better recognition
        service_mappings = {
            # Messaging & Communication
            'telegram': 'Telegram',
            'telegramTrigger': 'Telegram',
            'discord': 'Discord',
            'slack': 'Slack', 
            'whatsapp': 'WhatsApp',
            'mattermost': 'Mattermost',
            'teams': 'Microsoft Teams',
            'rocketchat': 'Rocket.Chat',
            
            # Email
            'gmail': 'Gmail',
            'mailjet': 'Mailjet',
            'emailreadimap': 'Email (IMAP)',
            'emailsendsmt': 'Email (SMTP)',
            'outlook': 'Outlook',
            
            # Cloud Storage
            'googledrive': 'Google Drive',
            'googledocs': 'Google Docs',
            'googlesheets': 'Google Sheets',
            'dropbox': 'Dropbox',
            'onedrive': 'OneDrive',
            'box': 'Box',
            
            # Databases
            'postgres': 'PostgreSQL',
            'mysql': 'MySQL',
            'mongodb': 'MongoDB',
            'redis': 'Redis',
            'airtable': 'Airtable',
            'notion': 'Notion',
            
            # Project Management
            'jira': 'Jira',
            'github': 'GitHub',
            'gitlab': 'GitLab',
            'trello': 'Trello',
            'asana': 'Asana',
            'mondaycom': 'Monday.com',
            
            # AI/ML Services
            'openai': 'OpenAI',
            'anthropic': 'Anthropic',
            'huggingface': 'Hugging Face',
            
            # Social Media
            'linkedin': 'LinkedIn',
            'twitter': 'Twitter/X',
            'facebook': 'Facebook',
            'instagram': 'Instagram',
            
            # E-commerce
            'shopify': 'Shopify',
            'stripe': 'Stripe',
            'paypal': 'PayPal',
            
            # Analytics
            'googleanalytics': 'Google Analytics',
            'mixpanel': 'Mixpanel',
            
            # Calendar & Tasks
            'googlecalendar': 'Google Calendar', 
            'googletasks': 'Google Tasks',
            'cal': 'Cal.com',
            'calendly': 'Calendly',
            
            # Forms & Surveys
            'typeform': 'Typeform',
            'googleforms': 'Google Forms',
            'form': 'Form Trigger',
            
            # Development Tools
            'webhook': 'Webhook',
            'httpRequest': 'HTTP Request',
            'graphql': 'GraphQL',
            'sse': 'Server-Sent Events',
            
            # Utility nodes (exclude from integrations)
            'set': None,
            'function': None,
            'code': None,
            'if': None,
            'switch': None,
            'merge': None,
            'split': None,
            'stickynote': None,
            'stickyNote': None,
            'wait': None,
            'schedule': None,
            'cron': None,
            'manual': None,
            'stopanderror': None,
            'noop': None,
            'noOp': None,
            'error': None,
            'limit': None,
            'aggregate': None,
            'summarize': None,
            'filter': None,
            'sort': None,
            'removeDuplicates': None,
            'dateTime': None,
            'extractFromFile': None,
            'convertToFile': None,
            'readBinaryFile': None,
            'readBinaryFiles': None,
            'executionData': None,
            'executeWorkflow': None,
            'executeCommand': None,
            'respondToWebhook': None,
        }
        
        for node in nodes:
            node_type = node.get('type', '')
            node_name = node.get('name', '').lower()
            
            # Determine trigger type
            if 'webhook' in node_type.lower() or 'webhook' in node_name:
                trigger_type = 'Webhook'
            elif 'cron' in node_type.lower() or 'schedule' in node_type.lower():
                trigger_type = 'Scheduled'
            elif 'trigger' in node_type.lower() and trigger_type == 'Manual':
                if 'manual' not in node_type.lower():
                    trigger_type = 'Webhook'
            
            # Extract integrations with enhanced mapping
            service_name = None
            
            # Handle n8n-nodes-base nodes
            if node_type.startswith('n8n-nodes-base.'):
                raw_service = node_type.replace('n8n-nodes-base.', '').lower()
                raw_service = raw_service.replace('trigger', '')
                service_name = service_mappings.get(raw_service, raw_service.title() if raw_service else None)
            
            # Handle @n8n/ namespaced nodes
            elif node_type.startswith('@n8n/'):
                raw_service = node_type.split('.')[-1].lower() if '.' in node_type else node_type.lower()
                raw_service = raw_service.replace('trigger', '')
                service_name = service_mappings.get(raw_service, raw_service.title() if raw_service else None)
            
            # Handle custom nodes
            elif '-' in node_type:
                # Try to extract service name from custom node names like "n8n-nodes-youtube-transcription-kasha.youtubeTranscripter"
                parts = node_type.lower().split('.')
                for part in parts:
                    if 'youtube' in part:
                        service_name = 'YouTube'
                        break
                    elif 'telegram' in part:
                        service_name = 'Telegram'
                        break
                    elif 'discord' in part:
                        service_name = 'Discord'
                        break
            
            # Also check node names for service hints
            for service_key, service_value in service_mappings.items():
                if service_key in node_name and service_value:
                    service_name = service_value
                    break
            
            # Add to integrations if valid service found
            if service_name and service_name not in ['None', None]:
                integrations.add(service_name)
        
        # Determine if complex based on node variety and count
        if len(nodes) > 10 and len(integrations) > 3:
            trigger_type = 'Complex'
        
        return trigger_type, integrations
    
    def generate_description(self, workflow: Dict, trigger_type: str, integrations: set) -> str:
        """Generate a descriptive summary of the workflow."""
        name = workflow['name']
        node_count = workflow['node_count']
        
        # Start with trigger description
        trigger_descriptions = {
            'Webhook': "Webhook-triggered automation that",
            'Scheduled': "Scheduled automation that", 
            'Complex': "Complex multi-step automation that",
        }
        desc = trigger_descriptions.get(trigger_type, "Manual workflow that")
        
        # Add functionality based on name and integrations
        if integrations:
            main_services = list(integrations)[:3]
            if len(main_services) == 1:
                desc += f" integrates with {main_services[0]}"
            elif len(main_services) == 2:
                desc += f" connects {main_services[0]} and {main_services[1]}"
            else:
                desc += f" orchestrates {', '.join(main_services[:-1])}, and {main_services[-1]}"
        
        # Add workflow purpose hints from name
        name_lower = name.lower()
        if 'create' in name_lower:
            desc += " to create new records"
        elif 'update' in name_lower:
            desc += " to update existing data"
        elif 'sync' in name_lower:
            desc += " to synchronize data"
        elif 'notification' in name_lower or 'alert' in name_lower:
            desc += " for notifications and alerts"
        elif 'backup' in name_lower:
            desc += " for data backup operations"
        elif 'monitor' in name_lower:
            desc += " for monitoring and reporting"
        else:
            desc += " for data processing"
        
        desc += f". Uses {node_count} nodes"
        if len(integrations) > 3:
            desc += f" and integrates with {len(integrations)} services"
        
        return desc + "."
    
    def index_all_workflows(self, force_reindex: bool = False, batch_size: int = 20) -> Dict[str, int]:
        """Index all workflows in the workflows directory with better error handling."""
        if not os.path.exists(self.workflows_dir):
            print(f"Workflows directory '{self.workflows_dir}' not found.")
            return {'processed': 0, 'skipped': 0, 'errors': 0}
            
        # Get list of JSON files
        json_files = []
        for root, _, files in os.walk(self.workflows_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        print(f"Indexing {len(json_files)} workflow files...")
        
        stats = {'processed': 0, 'skipped': 0, 'errors': 0}
        
        for i, file_path in enumerate(json_files, 1):
            try:
                # Get a fresh connection for each batch
                if i % batch_size == 1:
                    if i > 1:
                        try:
                            self.conn.commit()
                            self.conn.close()
                        except Exception as e:
                            print(f"Error closing connection: {e}")
                    self.conn = self.get_connection()
                
                # Check if we need to reindex
                try:
                    file_stat = os.stat(file_path)
                    file_hash = self._get_file_hash(file_path)
                    file_size = file_stat.st_size
                except Exception as e:
                    print(f"Error getting file info for {file_path}: {e}")
                    stats['errors'] += 1
                    continue
                
                cursor = self.conn.cursor()
                try:
                    cursor.execute(
                        "SELECT id, file_hash FROM workflows WHERE filename = ?",
                        (file_path,)
                    )
                    existing = cursor.fetchone()
                except Exception as e:
                    print(f"Error checking if workflow exists: {e}")
                    stats['errors'] += 1
                    continue
                
                if existing and not force_reindex and existing['file_hash'] == file_hash:
                    # Skip if file hasn't changed
                    stats['skipped'] += 1
                    if i % 50 == 0 or i == len(json_files):
                        print(f"Processed {i}/{len(json_files)} files... (Skipped: {stats['skipped']}, Errors: {stats['errors']})")
                    continue
                
                # Process the workflow file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        workflow_data = json.load(f)
                    
                    # Analyze workflow
                    workflow_info = self.analyze_workflow_file(file_path)
                    if not workflow_info:
                        print(f"Warning: Failed to analyze workflow {file_path}")
                        stats['errors'] += 1
                        continue
                    
                    # Update or insert workflow
                    try:
                        if existing:
                            self._update_workflow(self.conn, existing['id'], workflow_info, file_hash, file_size)
                        else:
                            self._insert_workflow(self.conn, workflow_info, file_path, file_hash, file_size)
                        
                        stats['processed'] += 1
                        
                        # Generate and store search text and embedding
                        search_text = self._generate_search_text(workflow_info)
                        embedding = self.generate_embedding(search_text)
                        
                        if embedding is not None:
                            cursor.execute(
                                "UPDATE workflows SET search_text = ?, embedding = ? WHERE filename = ?",
                                (search_text, sqlite3.Binary(embedding), file_path)
                            )
                    
                    except Exception as e:
                        print(f"Error updating database for {file_path}: {e}")
                        stats['errors'] += 1
                        continue
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {file_path}: {e}")
                    stats['errors'] += 1
                    continue
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    stats['errors'] += 1
                    continue
                
                # Commit after each batch
                if i % batch_size == 0 or i == len(json_files):
                    try:
                        self.conn.commit()
                        print(f"✅ Committed batch: Processed {i}/{len(json_files)} files... "
                              f"(Processed: {stats['processed']}, Skipped: {stats['skipped']}, Errors: {stats['errors']})")
                    except Exception as e:
                        print(f"Error committing batch: {e}")
                        # Reset connection on commit error
                        try:
                            self.conn.rollback()
                            self.conn.close()
                        except Exception as e:
                            print(f"Error during rollback/close: {e}")
                        self.conn = self.get_connection()
                
            except Exception as e:
                print(f"Unexpected error processing {file_path}: {e}")
                stats['errors'] += 1
                # Reset connection on error
                try:
                    self.conn.close()
                except Exception as e:
                    print(f"Error closing connection: {e}")
                self.conn = self.get_connection()
                continue
        
        # Final commit and cleanup
        try:
            if self.conn:
                self.conn.commit()
                self.conn.close()
                self.conn = None
        except Exception as e:
            print(f"Error during final commit/close: {e}")
        
        print(f"\n✅ Indexing complete! "
              f"Processed: {stats['processed']}, "
              f"Skipped: {stats['skipped']}, "
              f"Errors: {stats['errors']}")
        
        # Print summary
        if stats['errors'] > 0:
            print("\n⚠️  Some workflows had errors during indexing. Check the logs for details.")
        
        return stats
    
    def search_workflows(self, query: str = "", trigger_filter: str = "all", 
                        complexity_filter: str = "all", active_only: bool = False,
                        limit: int = 50, offset: int = 0) -> Tuple[List[Dict], int]:
        """Fast search with filters and pagination."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        
        # Build WHERE clause
        where_conditions = []
        params = []
        
        if active_only:
            where_conditions.append("w.active = 1")
        
        if trigger_filter != "all":
            where_conditions.append("w.trigger_type = ?")
            params.append(trigger_filter)
        
        if complexity_filter != "all":
            where_conditions.append("w.complexity = ?")
            params.append(complexity_filter)
        
        # Use FTS search if query provided
        if query.strip():
            # FTS search with ranking
            base_query = """
                SELECT w.*, rank
                FROM workflows_fts fts
                JOIN workflows w ON w.id = fts.rowid
                WHERE workflows_fts MATCH ?
            """
            params.insert(0, query)
        else:
            # Regular query without FTS
            base_query = """
                SELECT w.*, 0 as rank
                FROM workflows w
                WHERE 1=1
            """
        
        if where_conditions:
            base_query += " AND " + " AND ".join(where_conditions)
        
        # Count total results
        count_query = f"SELECT COUNT(*) as total FROM ({base_query}) t"
        cursor = conn.execute(count_query, params)
        total = cursor.fetchone()['total']
        
        # Get paginated results
        if query.strip():
            base_query += " ORDER BY rank"
        else:
            base_query += " ORDER BY w.analyzed_at DESC"
        
        base_query += f" LIMIT {limit} OFFSET {offset}"
        
        cursor = conn.execute(base_query, params)
        rows = cursor.fetchall()
        
        # Convert to dictionaries and parse JSON fields
        results = []
        for row in rows:
            workflow = dict(row)
            workflow['integrations'] = json.loads(workflow['integrations'] or '[]')
            
            # Parse tags and convert dict tags to strings
            raw_tags = json.loads(workflow['tags'] or '[]')
            clean_tags = []
            for tag in raw_tags:
                if isinstance(tag, dict):
                    # Extract name from tag dict if available
                    clean_tags.append(tag.get('name', str(tag.get('id', 'tag'))))
                else:
                    clean_tags.append(str(tag))
            workflow['tags'] = clean_tags
            
            results.append(workflow)
        
        conn.close()
        return results, total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        
        # Basic counts
        cursor = conn.execute("SELECT COUNT(*) as total FROM workflows")
        total = cursor.fetchone()['total']
        
        cursor = conn.execute("SELECT COUNT(*) as active FROM workflows WHERE active = 1")
        active = cursor.fetchone()['active']
        
        # Trigger type breakdown
        cursor = conn.execute("""
            SELECT trigger_type, COUNT(*) as count 
            FROM workflows 
            GROUP BY trigger_type
        """)
        triggers = {row['trigger_type']: row['count'] for row in cursor.fetchall()}
        
        # Complexity breakdown
        cursor = conn.execute("""
            SELECT complexity, COUNT(*) as count 
            FROM workflows 
            GROUP BY complexity
        """)
        complexity = {row['complexity']: row['count'] for row in cursor.fetchall()}
        
        # Node stats
        cursor = conn.execute("SELECT SUM(node_count) as total_nodes FROM workflows")
        total_nodes = cursor.fetchone()['total_nodes'] or 0
        
        # Unique integrations count
        cursor = conn.execute("SELECT integrations FROM workflows WHERE integrations != '[]'")
        all_integrations = set()
        for row in cursor.fetchall():
            integrations = json.loads(row['integrations'])
            all_integrations.update(integrations)
        
        conn.close()
        
        return {
            'total': total,
            'active': active,
            'inactive': total - active,
            'triggers': triggers,
            'complexity': complexity,
            'total_nodes': total_nodes,
            'unique_integrations': len(all_integrations),
            'last_indexed': datetime.datetime.now().isoformat()
        }

    def get_service_categories(self) -> Dict[str, List[str]]:
        """Get service categories for enhanced filtering."""
        return {
            'messaging': ['Telegram', 'Discord', 'Slack', 'WhatsApp', 'Mattermost', 'Microsoft Teams', 'Rocket.Chat'],
            'email': ['Gmail', 'Mailjet', 'Email (IMAP)', 'Email (SMTP)', 'Outlook'],
            'cloud_storage': ['Google Drive', 'Google Docs', 'Google Sheets', 'Dropbox', 'OneDrive', 'Box'],
            'database': ['PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Airtable', 'Notion'],
            'project_management': ['Jira', 'GitHub', 'GitLab', 'Trello', 'Asana', 'Monday.com'],
            'ai_ml': ['OpenAI', 'Anthropic', 'Hugging Face'],
            'social_media': ['LinkedIn', 'Twitter/X', 'Facebook', 'Instagram'],
            'ecommerce': ['Shopify', 'Stripe', 'PayPal'],
            'analytics': ['Google Analytics', 'Mixpanel'],
            'calendar_tasks': ['Google Calendar', 'Google Tasks', 'Cal.com', 'Calendly'],
            'forms': ['Typeform', 'Google Forms', 'Form Trigger'],
            'development': ['Webhook', 'HTTP Request', 'GraphQL', 'Server-Sent Events', 'YouTube']
        }

    def search_by_category(self, category: str, limit: int = 50, offset: int = 0) -> Tuple[List[Dict], int]:
        """Search workflows by service category."""
        categories = self.get_service_categories()
        if category not in categories:
            return [], 0
        
        services = categories[category]
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Build OR conditions for all services in category
        service_conditions = []
        params = []
        for service in services:
            service_conditions.append("integrations LIKE ?")
            params.append(f'%"{service}"%')
        
        where_clause = " OR ".join(service_conditions)
        
        # Count total results
        count_query = f"SELECT COUNT(*) as total FROM workflows WHERE {where_clause}"
        cursor = conn.execute(count_query, params)
        total = cursor.fetchone()['total']
        
        # Get paginated results
        query = f"""
            SELECT * FROM workflows 
            WHERE {where_clause}
            ORDER BY analyzed_at DESC
            LIMIT {limit} OFFSET {offset}
        """
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to dictionaries and parse JSON fields
        results = []
        for row in rows:
            workflow = dict(row)
            workflow['integrations'] = json.loads(workflow['integrations'] or '[]')
            raw_tags = json.loads(workflow['tags'] or '[]')
            clean_tags = []
            for tag in raw_tags:
                if isinstance(tag, dict):
                    clean_tags.append(tag.get('name', str(tag.get('id', 'tag'))))
                else:
                    clean_tags.append(str(tag))
            workflow['tags'] = clean_tags
            results.append(workflow)
        
        conn.close()
        return results, total


def main():
    """Command-line interface for workflow database."""
    import argparse
    
    parser = argparse.ArgumentParser(description='N8N Workflow Database')
    parser.add_argument('--index', action='store_true', help='Index all workflows')
    parser.add_argument('--force', action='store_true', help='Force reindex all files')
    parser.add_argument('--search', help='Search workflows')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    
    args = parser.parse_args()
    
    db = WorkflowDatabase()
    
    if args.index:
        stats = db.index_all_workflows(force_reindex=args.force)
        print(f"Indexed {stats['processed']} workflows")
    
    elif args.search:
        results, total = db.search_workflows(args.search, limit=10)
        print(f"Found {total} workflows:")
        for workflow in results:
            print(f"  - {workflow['name']} ({workflow['trigger_type']}, {workflow['node_count']} nodes)")
    
    elif args.stats:
        stats = db.get_stats()
        print(f"Database Statistics:")
        print(f"  Total workflows: {stats['total']}")
        print(f"  Active: {stats['active']}")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Unique integrations: {stats['unique_integrations']}")
        print(f"  Trigger types: {stats['triggers']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()