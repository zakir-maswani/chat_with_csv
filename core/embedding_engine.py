import openai
import tiktoken
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid
import logging
import json
from datetime import datetime
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.settings import (
    CAR_AI_QDRANT_API_KEY,
    CAR_AI_QDRANT_URL
)
from utils.logger_setup import setup_logging, log_and_print, Colors

@dataclass
class TextDocument:
    """Represents a complete text document with metadata"""
    id: str
    text: str
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    created_at: str = None
    

class CompleteTextEmbedder:
    """
    Embedder for complete text documents without chunking.
    Embeds entire documents and stores them in Qdrant collections.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 embedding_model: str = "text-embedding-3-small",
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None,
                 qdrant_port: Optional[int] = None,
                 log_level: int = logging.INFO,
                 max_token_limit: int = 8000,
                 system_logger: Optional[logging.Logger] = None):
        """
        Initialize the complete text embedder
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            embedding_model: OpenAI embedding model to use
            qdrant_url: Qdrant Cloud URL or local host
            qdrant_api_key: Qdrant Cloud API key (optional for local)
            qdrant_port: Qdrant server port (optional for cloud)
            log_level: Logging level
            max_token_limit: Maximum tokens allowed per document
        """
        # Configure logging
        self.logger = system_logger if system_logger else setup_logging()
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_token_limit = max_token_limit
        
        # Initialize Qdrant client
        try:
            if qdrant_api_key:
                # Qdrant Cloud configuration
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=60
                )
                log_and_print(self.logger, 'DEBUG', f"Connected to Qdrant Cloud at: {qdrant_url}", Colors.CYAN)
            else:
                # Local Qdrant configuration
                self.qdrant_client = QdrantClient(
                    url=qdrant_url or "localhost",
                    port=qdrant_port or 6333,
                    timeout=60
                )
                log_and_print(self.logger, 'DEBUG', f"Connected to local Qdrant at: {qdrant_url or 'localhost'}:{qdrant_port or 6333}", Colors.CYAN)
                
            # Test connection
            collections = self.qdrant_client.get_collections()
            log_and_print(self.logger, 'INFO', f"Successfully connected! Found {len(collections.collections)} existing collections.", Colors.GREEN)
            
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Failed to connect to Qdrant: {e}", Colors.RED)
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            log_and_print(self.logger, 'DEBUG', f"Generated embedding of size {len(embedding)} for text length: {len(text)} chars", Colors.CYAN)
            return embedding
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Error getting embedding: {e}", Colors.RED)
            return []
    
    def create_collection(self, collection_name: str, db_config: Optional[Dict[str, Any]] = None, vector_size: int = 1536) -> bool:
        """Create a Qdrant collection for storing embeddings"""
        try:
            # Use existing client if no db_config provided
            if not db_config:
                qdrant_client = self.qdrant_client
            else:
                # Initialize new client with provided configuration
                qdrant_client = QdrantClient(
                    url=db_config.get('SHOP_QDRANT_URL', CAR_AI_QDRANT_URL),
                    api_key=db_config.get('SHOP_QDRANT_API_KEY', CAR_AI_QDRANT_API_KEY),
                    timeout=60
                )

            # Check if collection already exists
            existing_collections = qdrant_client.get_collections()
            collection_names = [col.name for col in existing_collections.collections]
            
            if collection_name in collection_names:
                log_and_print(self.logger, 'DEBUG', f"Collection '{collection_name}' already exists", Colors.CYAN)
                return True
            
            # Create new collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            log_and_print(self.logger, 'INFO', f"Successfully created collection: '{collection_name}'", Colors.GREEN)
            
            # Verify collection was created
            updated_collections = qdrant_client.get_collections()
            updated_collection_names = [col.name for col in updated_collections.collections]
            
            return collection_name in updated_collection_names
            
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Error creating collection '{collection_name}': {e}", Colors.RED)
            raise
    
    def validate_text(self, text: str) -> tuple[bool, str]:
        """
        Validate and process text before processing
        
        Returns:
            tuple: (is_valid, processed_text_or_error_message)
        """
        if not text or not text.strip():
            return False, "Text is empty or contains only whitespace"
        
        token_count = self.count_tokens(text)
        if token_count > self.max_token_limit:
            # Trim text to fit within token limit
            trimmed_text = self.trim_to_token_limit(text)
            return True, trimmed_text
        
        return True, text

    def trim_to_token_limit(self, text: str) -> str:
        """
        Trim text to fit within the maximum token limit
        
        Args:
            text: Input text to trim
            
        Returns:
            str: Trimmed text that fits within token limit
        """
        # Simple approach: trim by estimated character ratio
        # More sophisticated approach would use actual tokenization
        estimated_chars_per_token = len(text) / self.count_tokens(text)
        target_chars = int(self.max_token_limit * estimated_chars_per_token * 0.9)  # 90% safety margin
        
        if len(text) <= target_chars:
            return text
        
        # Trim and try to end at a sentence or word boundary
        trimmed = text[:target_chars]
        
        # Try to end at sentence boundary
        last_period = trimmed.rfind('.')
        if last_period > target_chars * 0.8:  # If sentence ending is reasonably close
            return trimmed[:last_period + 1]
        
        # Try to end at word boundary
        last_space = trimmed.rfind(' ')
        if last_space > target_chars * 0.8:  # If word boundary is reasonably close
            return trimmed[:last_space]
        
        return trimmed
    
    def create_document(self, text: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> TextDocument:
        """
        Create a TextDocument from input text
        
        Args:
            text: Input text to embed
            metadata: Optional metadata dictionary
            doc_id: Optional custom document ID
            
        Returns:
            TextDocument object
        """
        # Validate text
        is_valid, error_msg = self.validate_text(text)
        if not is_valid:
            raise ValueError(f"Invalid text: {error_msg}")
        is_valid, result = self.validate_text(text)
        if not is_valid:
            raise ValueError(f"Invalid text: {result}")
        else:
            # Use the processed (potentially trimmed) text
            processed_text = result
        # Generate ID if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        # Count tokens
        token_count = self.count_tokens(text)
        
        # Create document
        document = TextDocument(
            id=doc_id,
            text=processed_text.strip(),
            token_count=token_count,
            metadata=metadata or {},
            created_at=datetime.now().isoformat()
        )
        
        log_and_print(self.logger, 'DEBUG', f"Created document {doc_id} with {token_count} tokens")
        return document
    
    def embed_document(self, document: TextDocument) -> TextDocument:
        """
        Generate embedding for a document

        Args:
            document: TextDocument to embed

        Returns:
            TextDocument with embedding added
        """
        try:
            # Safely get tags from metadata and convert to string
            tags_str = ""
            if document.metadata and "tags" in document.metadata:
                tags = document.metadata["tags"]
                if isinstance(tags, list):
                    tags_str = " ".join(str(tag) for tag in tags)
                else:
                    tags_str = str(tags)
                tags_str = f"\nTAGS: {tags_str}"

            embedding_input = document.text + tags_str
            embedding = self.get_embedding(embedding_input)
            document.embedding = embedding

            if embedding:
                log_and_print(self.logger, 'DEBUG', f"Generated embedding for document {document.id}")
                # Log embedding statistics
                emb_array = np.array(embedding)
                log_and_print(self.logger, 'DEBUG', f"Embedding stats - Min: {emb_array.min():.4f}, "
                                  f"Max: {emb_array.max():.4f}, Mean: {emb_array.mean():.4f}")
            else:
                log_and_print(self.logger, 'WARNING', f"Failed to generate embedding for document {document.id}", Colors.YELLOW)

            return document

        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Error embedding document {document.id}: {e}", Colors.RED)
            raise

    def save_to_qdrant(self, documents: List[TextDocument], collection_name: str, db_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save documents with embeddings to Qdrant
        
        Args:
            documents: List of TextDocument objects with embeddings
            collection_name: Name of the Qdrant collection
            db_config: Optional dictionary containing database configuration
        
        Returns:
            bool: Success status
        """
        if not documents:
            log_and_print(self.logger, 'WARNING', "No documents to save", Colors.YELLOW)
            return False

        