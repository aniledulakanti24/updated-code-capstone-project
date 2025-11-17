from datetime import datetime
from typing import List, Dict, Any


class MemoryBank:
    """Long-term memory storage for learning from past detections"""

    def __init__(self):
        self.short_term: List[Dict] = []  # Recent detections
        self.long_term: Dict[str, List[Dict]] = {}  # Categorized patterns
        self.knowledge_base: Dict[str, Any] = {}  # Learned rules

    def add_to_short_term(self, memory: Dict):
        self.short_term.append(memory)
        if len(self.short_term) > 100:  # Keep last 100
            self.consolidate_memory()

    def consolidate_memory(self):
        """Move short-term to long-term memory"""
        for item in self.short_term[:50]:
            category = item.get("category", "unknown")
            if category not in self.long_term:
                self.long_term[category] = []
            self.long_term[category].append(item)
        self.short_term = self.short_term[50:]

    def learn_pattern(self, pattern_type: str, pattern: str, confidence: float):
        """Store learned patterns for future detection"""
        if pattern_type not in self.knowledge_base:
            self.knowledge_base[pattern_type] = []
        self.knowledge_base[pattern_type].append({
            "pattern": pattern,
            "confidence": confidence,
            "learned_at": datetime.now().isoformat()
        })

    def get_relevant_memory(self, context: str) -> List[Dict]:
        """Retrieve relevant memories for context"""
        relevant = []
        for category, memories in self.long_term.items():
            if context.lower() in category.lower():
                relevant.extend(memories[-5:])  # Last 5 relevant
        return relevant
