try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None
import faiss
import numpy as np
import os
from datetime import datetime
from typing import List, Optional, Dict
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
import json

logger = logging.getLogger(__name__)

class Vectorizer:
    """Wrapper for sentence transformer with similarity search"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            logger.warning("sentence_transformers not available; using fallback vectorizer")
            self.model = None
        else:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer: {e}")
                self.model = None

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model:
            # very small fallback embedding using bag-of-words length
            return np.array([len(text)])
        return self.model.encode(text, convert_to_tensor=False, normalize_embeddings=True)

    def find_similar(self, query: str, entries: List[Dict], top_k: int = 3) -> List[Dict]:
        """Find most similar entries using cosine similarity"""
        if not entries:
            return []
            
        try:
            # Get query embedding
            query_vec = self.embed(query)

            if not self.model:
                # naive similarity by common words length
                query_terms = set(query.lower().split())
                scored = []
                for e in entries:
                    text = f"{e['goal']} {e.get('result', '')}".lower()
                    score = len(query_terms & set(text.split()))
                    scored.append((score, e))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [e for s, e in scored[:top_k] if s > 0]

            # Get embeddings for all entries
            entry_texts = [f"{e['goal']} {e.get('result', '')}" for e in entries]
            entry_vecs = self.model.encode(entry_texts, convert_to_tensor=False, normalize_embeddings=True)

            # Compute similarities
            similarities = np.dot(entry_vecs, query_vec)

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Return entries sorted by similarity
            return [entries[i] for i in top_indices]

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

def get_vectorizer() -> Optional[Vectorizer]:
    """Initialize vectorizer safely"""
    return Vectorizer()

class VectorMemory:
    # tests call VectorMemory(filename="…")
    def __init__(self, filename: str | None = None, **kwargs):
        """
        Very light stub – enough to satisfy tests that perform a .search().
        """
        self.path = Path(filename or kwargs.get("file_path", "vector.db"))
        self._index: list[tuple[str, str]] = []
        
        try:
            vec = get_vectorizer()
            if vec and getattr(vec, "model", None) is not None:
                self.vectorizer = vec
            else:
                self.vectorizer = None
                logger.warning("Running without vector search capability")
        except Exception as e:
            logger.warning(f"Vector initialization failed: {e}")
            self.vectorizer = None
            
        self.load_entries()
        
    def load_entries(self) -> None:
        """Load entries from file"""
        if not self.path.exists():
            return
            
        self.entries = []
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        self.entries.append(entry)
                    except json.JSONDecodeError:
                        continue
                        
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar entries"""
        if not self.vectorizer:
            # Fallback to basic keyword matching
            matches = []
            query_terms = set(query.lower().split())
            for entry in self.entries:
                entry_terms = set(entry["goal"].lower().split())
                if query_terms & entry_terms:
                    matches.append(entry)
            return sorted(matches, key=lambda x: len(set(x["goal"].lower().split()) & query_terms), reverse=True)[:top_k]
            
        return self.vectorizer.find_similar(query, self.entries, top_k)
        
    def add(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Add new entry"""
        entry = {
            "goal": text,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            entry.update(metadata)
            
        self.entries.append(entry)
        
        # Save to file
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        
    def encode(self, text: str) -> Optional[np.ndarray]:
        """Safely encode text to vector"""
        if not self.vectorizer:
            return None
            
        try:
            return self.vectorizer.embed(text)
        except Exception as e:
            logger.warning(f"Encoding failed: {e}")
            return None
            
    def add_entry(self, text: str, metadata: dict) -> bool:
        """Add entry with vector encoding"""
        vector = self.encode(text)
        if vector is None:
            return False
            
        self.vectors = vector
        self.entries.append(metadata)
        return True
        
    def add(self, text: str):
        """Add text to vector memory with timestamp"""
        # Add timestamp to text
        timestamp = datetime.now().isoformat()
        text_with_time = f"[{timestamp}] {text}"
        
        # Create and add embedding
        embedding = self.encode(text)
        if embedding is None:
            return
        self.vectors = embedding
        self.entries.append({
            "goal": text,
            "timestamp": timestamp
        })
        
    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.6
    ) -> List[dict]:
        """Find similar entries using vector similarity"""
        if not self.vectorizer or not self.vectors:
            return []
            
        query_vector = self.encode(query)
        if query_vector is None:
            return []
            
        # Calculate cosine similarities
        similarities = np.dot(self.vectors, query_vector)
        
        # Get top matches above threshold
        indices = (-similarities).argsort()[:top_k]
        matches = []
        
        for idx in indices:
            score = similarities[idx]
            if score < threshold:
                break
            matches.append({
                **self.entries[idx],
                "similarity": float(score)
            })
            
        return matches
        
    def save(self, path: str = "vector_memory"):
        """Save vector memory to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, f"{path}/index.faiss")
        
        # Save texts and timestamps
        with open(f"{path}/memory.jsonl", "w") as f:
            for text, timestamp in zip(self.texts, self.timestamps):
                f.write(f"{text}\t{timestamp}\n")
                
    def load(self, path: str = "vector_memory"):
        """Load vector memory from disk"""
        if not os.path.exists(path):
            return
            
        # Load index
        self.index = faiss.read_index(f"{path}/index.faiss")
        
        # Load texts and timestamps
        self.texts = []
        self.timestamps = []
        with open(f"{path}/memory.jsonl", "r") as f:
            for line in f:
                text, timestamp = line.strip().split("\t")
                self.texts.append(text)
                self.timestamps.append(timestamp) 