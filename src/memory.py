"""
SSM Framework - Memory Management Module
========================================

This module provides persistent storage and episodic memory capabilities
for the SSM framework. It integrates ChromaDB as a vector store for
semantic retrieval of past conversation segments and maintains the
deterministic JSON State Map as the primary source of truth.

Key Components:
- StateMapManager: Handles persistence of the GlobalStateMap to disk
- EpisodicMemoryStore: ChromaDB-based semantic memory for conversation history
- ConversationTurn: Structured representation of a single conversation turn
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema import GlobalStateMap, ExtractionResult, ConflictRecord


class ConversationTurn(BaseModel):
    """
    A structured record of a single conversation turn.
    
    Captures all relevant information from an interaction including
    user input, extracted state, conflicts, resolutions, and responses.
    Tracks chain-of-thought reasoning for debugging and improvement.
    """
    turn_id: int = Field(description="Sequential turn number")
    timestamp: datetime = Field(default_factory=datetime.now)
    user_input: str = Field(description="User's input message")
    extraction: Optional[ExtractionResult] = Field(
        default=None,
        description="Extracted state from this turn (includes chain_of_thought)"
    )
    conflicts: List[ConflictRecord] = Field(
        default_factory=list,
        description="Conflicts detected in this turn"
    )
    conflict_resolution_choice: Optional[str] = Field(
        default=None,
        description="User's choice when conflicts detected: 'accept' (merge), 'reject' (keep original), or None"
    )
    merged_fields: List[str] = Field(
        default_factory=list,
        description="Which fields were actually merged into state after conflict resolution"
    )
    response: str = Field(default="", description="System response to user")
    embedding_text: str = Field(
        default="",
        description="Text used for generating embedding"
    )
    langsmith_run_id: Optional[str] = Field(
        default=None,
        description="LangSmith run ID for this turn"
    )
    
    def to_embedding_text(self) -> str:
        """Generate text suitable for semantic embedding."""
        parts = [
            f"User: {self.user_input}",
            f"Response: {self.response}"
        ]
        if self.extraction:
            parts.append(f"Intent: {self.extraction.raw_intent}")
        return " | ".join(parts)


class StateMapManager:
    """
    Manages persistence of the GlobalStateMap to disk.
    
    This class provides a deterministic, JSON-based persistence layer
    for the state map, ensuring that state survives across sessions
    and can be audited or restored at any time.
    """
    
    def __init__(
        self,
        persist_dir: str = "./data/state",
        state_filename: str = "global_state_map.json"
    ):
        """
        Initialize the State Map Manager.
        
        Args:
            persist_dir: Directory for storing state files
            state_filename: Name of the main state file
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.persist_dir / state_filename
        self.backup_dir = self.persist_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load or create the state map
        self._state_map: Optional[GlobalStateMap] = None
    
    @property
    def state_map(self) -> GlobalStateMap:
        """Get the current state map, loading from disk if necessary."""
        if self._state_map is None:
            self._state_map = self.load()
        return self._state_map
    
    def load(self) -> GlobalStateMap:
        """
        Load the state map from disk.
        
        Returns the saved state map if it exists, otherwise creates
        and returns a new default state map.
        """
        if self.state_file.exists():
            try:
                return GlobalStateMap.from_json_file(str(self.state_file))
            except Exception as e:
                print(f"Warning: Failed to load state map: {e}")
                print("Creating new default state map.")
        
        return GlobalStateMap()
    
    def save(self, state_map: Optional[GlobalStateMap] = None) -> None:
        """
        Save the state map to disk.
        
        Creates a backup of the previous state before saving the new one.
        
        Args:
            state_map: State map to save (uses current if not provided)
        """
        if state_map is not None:
            self._state_map = state_map
        
        if self._state_map is None:
            return
        
        # Create backup of existing state
        if self.state_file.exists():
            backup_name = f"state_v{self._state_map.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = self.backup_dir / backup_name
            try:
                import shutil
                shutil.copy2(str(self.state_file), str(backup_path))
            except Exception as e:
                print(f"Warning: Could not create backup: {e}")
        
        # Save current state
        self._state_map.to_json_file(str(self.state_file))
    
    def reset(self) -> GlobalStateMap:
        """
        Reset the state map to default values.
        
        Backs up the current state, then clears it entirely.
        On next load, a completely fresh default state is created.
        
        Returns:
            A new default state map
        """
        if self._state_map is not None:
            backup_name = f"state_reset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = self.backup_dir / backup_name
            self._state_map.to_json_file(str(backup_path))
            print(f"[RESET] Backed up previous state to: {backup_path}")
        
        # Delete the main state file completely so it starts fresh next load
        if self.state_file.exists():
            self.state_file.unlink()
            print(f"[RESET] Deleted state file: {self.state_file}")
        
        # Create a fresh empty state
        self._state_map = GlobalStateMap()
        print(f"[RESET] Created new empty state map")
        
        # Don't save yet - let next load create a fresh file
        return self._state_map
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get a list of available backup versions.
        
        Returns:
            List of backup file information
        """
        backups = []
        for backup_file in self.backup_dir.glob("*.json"):
            try:
                stat = backup_file.stat()
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception:
                continue
        
        return sorted(backups, key=lambda x: x["modified"], reverse=True)


class EpisodicMemoryStore:
    """
    ChromaDB-based episodic memory store for conversation history.
    
    This class provides semantic search capabilities over past conversation
    turns, enabling the system to retrieve relevant context from previous
    interactions based on semantic similarity rather than exact matches.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "conversation_history",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the Episodic Memory Store.
        
        Args:
            persist_directory: Directory for ChromaDB storage
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the sentence transformer model
        """
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Use sentence transformers for embeddings
        # Fall back to default if sentence-transformers not available
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        except Exception:
            print("Warning: SentenceTransformer not available, using default embedding")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_turn(
        self,
        turn: ConversationTurn,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to the memory store.
        
        Args:
            turn: The conversation turn to store
            metadata: Additional metadata to store with the turn
        """
        embedding_text = turn.to_embedding_text()
        
        doc_metadata = {
            "turn_id": turn.turn_id,
            "timestamp": turn.timestamp.isoformat(),
            "had_conflicts": len(turn.conflicts) > 0,
            "extraction_confidence": turn.extraction.confidence if turn.extraction else 0.0,
            "langsmith_run_id": turn.langsmith_run_id or ""
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        self.collection.add(
            documents=[embedding_text],
            metadatas=[doc_metadata],
            ids=[f"turn_{turn.turn_id}"]
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar conversation turns.
        
        Args:
            query: The search query
            n_results: Maximum number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of matching conversation turns with similarity scores
        """
        query_params = {
            "query_texts": [query],
            "n_results": n_results
        }
        
        if where_filter:
            query_params["where"] = where_filter
        
        results = self.collection.query(**query_params)
        
        formatted_results = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                result = {
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "id": results["ids"][0][i] if results["ids"] else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_recent_turns(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent conversation turns.
        
        Args:
            n: Number of recent turns to retrieve
            
        Returns:
            List of recent conversation turns
        """
        results = self.collection.get(
            limit=n,
            include=["documents", "metadatas"]
        )
        
        turns = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                turns.append({
                    "document": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })
        
        return turns
    
    def get_turns_with_conflicts(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent turns that had conflicts.
        
        Args:
            n: Maximum number of turns to retrieve
            
        Returns:
            List of turns that had detected conflicts
        """
        return self.search(
            query="conflict contradiction violation",
            n_results=n,
            where_filter={"had_conflicts": True}
        )
    
    def clear(self) -> None:
        """Clear all stored conversation history."""
        # Get all IDs and delete them
        all_items = self.collection.get()
        if all_items["ids"]:
            self.collection.delete(ids=all_items["ids"])
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory store.
        
        Returns:
            Dictionary containing memory store statistics
        """
        count = self.collection.count()
        
        return {
            "total_turns": count,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection.name
        }


class MemoryManager:
    """
    Unified memory management facade.
    
    Provides a single interface for both state map persistence and
    episodic memory operations, coordinating between the two systems.
    """
    
    def __init__(
        self,
        state_dir: str = "./data/state",
        chroma_dir: str = "./data/chroma_db",
        session_id: Optional[str] = None
    ):
        """
        Initialize the Memory Manager.
        
        Args:
            state_dir: Directory for state map persistence
            chroma_dir: Directory for ChromaDB storage
            session_id: Optional session identifier
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.state_manager = StateMapManager(
            persist_dir=state_dir
        )
        
        self.episodic_store = EpisodicMemoryStore(
            persist_directory=chroma_dir,
            collection_name=f"session_{self.session_id}"
        )
        
        self._turn_counter = 0
    
    @property
    def state_map(self) -> GlobalStateMap:
        """Get the current global state map."""
        return self.state_manager.state_map
    
    def save_state(self, state_map: Optional[GlobalStateMap] = None) -> None:
        """Save the state map to disk."""
        self.state_manager.save(state_map)
    
    def record_turn(
        self,
        user_input: str,
        extraction: Optional[ExtractionResult],
        conflicts: List[ConflictRecord],
        response: str,
        langsmith_run_id: Optional[str] = None
    ) -> ConversationTurn:
        """
        Record a complete conversation turn.
        
        Args:
            user_input: User's input message
            extraction: Result of state extraction
            conflicts: List of detected conflicts
            response: System response
            langsmith_run_id: LangSmith run ID for tracing
            
        Returns:
            The created ConversationTurn
        """
        self._turn_counter += 1
        
        turn = ConversationTurn(
            turn_id=self._turn_counter,
            user_input=user_input,
            extraction=extraction,
            conflicts=conflicts,
            response=response,
            langsmith_run_id=langsmith_run_id
        )
        
        self.episodic_store.add_turn(turn)
        
        return turn
    
    def find_relevant_context(
        self,
        query: str,
        n_results: int = 3
    ) -> List[str]:
        """
        Find relevant past context for a query.
        
        Args:
            query: The current query to find context for
            n_results: Number of relevant turns to retrieve
            
        Returns:
            List of relevant conversation excerpts
        """
        results = self.episodic_store.search(query, n_results=n_results)
        return [r["document"] for r in results]
    
    def reset_session(self) -> GlobalStateMap:
        """
        Reset the current session.
        
        Clears episodic memory and resets state map.
        
        Returns:
            A new default state map
        """
        self.episodic_store.clear()
        self._turn_counter = 0
        return self.state_manager.reset()
    
    def get_full_context(self) -> Dict[str, Any]:
        """
        Get complete context for the current session.
        
        Returns:
            Dictionary containing state map and recent history
        """
        return {
            "state_map": self.state_map.model_dump(),
            "turn_count": self._turn_counter,
            "episodic_stats": self.episodic_store.get_stats()
        }


# Export all classes
__all__ = [
    'ConversationTurn',
    'StateMapManager',
    'EpisodicMemoryStore',
    'MemoryManager',
]