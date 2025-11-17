from typing import List, Dict, Any


class ContextCompactor:
    """Compact and optimize context for efficient processing"""

    @staticmethod
    def compact_memory(memory: List[Dict], max_items: int = 10) -> List[Dict]:
        """Keep only most relevant recent memories"""
        if len(memory) <= max_items:
            return memory

        # Prioritize: most recent + highest confidence detections
        sorted_memory = sorted(
            memory, 
            key=lambda x: (x.get("confidence", 0), x.get("timestamp", "")),
            reverse=True
        )
        return sorted_memory[:max_items]

    @staticmethod
    def summarize_context(context: Dict) -> Dict:
        """Summarize large context objects"""
        summary = {}
        for key, value in context.items():
            if isinstance(value, list) and len(value) > 5:
                summary[key] = {
                    "count": len(value),
                    "sample": value[:3],
                    "summary": "truncated"
                }
            else:
                summary[key] = value
        return summary
