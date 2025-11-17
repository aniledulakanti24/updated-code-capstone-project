from datetime import datetime
from typing import List, Dict, Any, Optional


class MetricsCollector:
    """Collect and track system metrics"""

    def __init__(self):
        self.metrics = {
            "total_scans": 0,
            "total_detections": 0,
            "false_positives": 0,
            "true_positives": 0,
            "processing_time": [],
            "agent_calls": {}
        }

    def record_scan(self, duration: float):
        self.metrics["total_scans"] += 1
        self.metrics["processing_time"].append(duration)

    def record_detection(self, is_true_positive: bool):
        self.metrics["total_detections"] += 1
        if is_true_positive:
            self.metrics["true_positives"] += 1
        else:
            self.metrics["false_positives"] += 1

    def record_agent_call(self, agent_name: str):
        self.metrics["agent_calls"][agent_name] = \
            self.metrics["agent_calls"].get(agent_name, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        avg_time = sum(self.metrics["processing_time"]) / len(self.metrics["processing_time"]) \
            if self.metrics["processing_time"] else 0
        return {
            **self.metrics,
            "average_processing_time": avg_time,
            "accuracy": self.metrics["true_positives"] / max(self.metrics["total_detections"], 1)
        }


class TracingService:
    """Distributed tracing for agent operations"""

    def __init__(self):
        self.traces: List[Dict] = []

    def start_trace(self, operation: str, parent_id: Optional[str] = None) -> str:
        trace_id = f"trace_{len(self.traces)}_{datetime.now().timestamp()}"
        trace = {
            "trace_id": trace_id,
            "parent_id": parent_id,
            "operation": operation,
            "start_time": datetime.now(),
            "end_time": None,
            "status": "running",
            "metadata": {}
        }
        self.traces.append(trace)
        return trace_id

    def end_trace(self, trace_id: str, status: str = "success", metadata: Dict = None):
        for trace in self.traces:
            if trace["trace_id"] == trace_id:
                trace["end_time"] = datetime.now()
                trace["status"] = status
                if metadata:
                    trace["metadata"].update(metadata)
                break

    def get_trace_tree(self) -> List[Dict]:
        """Build hierarchical trace tree"""
        return [t for t in self.traces if t["parent_id"] is None]
