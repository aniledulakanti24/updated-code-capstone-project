import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

from .sessions import InMemorySessionService, SessionState, SessionData
from .memory import MemoryBank
from .observability import MetricsCollector, TracingService
from .tools import WebSearchTool, CodeExecutionTool, MCPTool, OpenAPITool
from .agents import DataCollectorAgent, DetectionAgent, LLMAgent, ActionAgent, LoopAgent
from .orchestrator import AgentOrchestrator
from .a2a import A2AProtocol, A2AMessage
from .evaluation import AgentEvaluator
from .context import ContextCompactor
from .deployment import DeploymentConfig, DeploymentManager


class DLPSystem:
    """Complete Data Leakage Prevention System"""

    def __init__(self):
        # Initialize all components
        self.session_service = InMemorySessionService()
        self.memory_bank = MemoryBank()
        self.metrics = MetricsCollector()
        self.tracer = TracingService()
        self.a2a_protocol = A2AProtocol()
        self.evaluator = AgentEvaluator()
        self.context_compactor = ContextCompactor()

        # Initialize tools
        self.tools = {
            "web_search": WebSearchTool(),
            "code_exec": CodeExecutionTool(),
            "mcp": MCPTool("SecurityService", "https://api.security.com"),
            "api": OpenAPITool({"version": "1.0"})
        }

        # Initialize agents
        self.agents = {
            "collector": DataCollectorAgent("collector"),
            "detector": DetectionAgent("detector", [self.tools["web_search"]]),
            "llm_analyzer": LLMAgent("llm_analyzer"),
            "action": ActionAgent("action"),
            "monitor": LoopAgent("monitor", max_iterations=5)
        }

        self.orchestrator = AgentOrchestrator(
            self.session_service, 
            self.memory_bank,
            self.metrics,
            self.tracer
        )

        self.logger = logging.getLogger("DLPSystem")
        logging.basicConfig(level=logging.INFO)

    async def scan_data(self, data: Dict, session_id: str = None) -> Dict[str, Any]:
        """Main entry point for scanning data"""
        if not session_id:
            session_id = f"session_{datetime.now().timestamp()}"

        # Create or get session
        session = self.session_service.get_session(session_id)
        if not session:
            session = self.session_service.create_session(session_id)

        start_time = datetime.now()

        # Step 1: Data Collection (if needed)
        if "sources" in data:
            collection_result = await self.agents["collector"].execute(
                data, {"memory": session.memory, "context": session.context}
            )
            data = collection_result["collected_data"][0]  # Use first source

        # Step 2: Parallel Detection
        detection_agents = [
            self.agents["detector"],
            self.agents["llm_analyzer"]
        ]

        detection_results = await self.orchestrator.run_parallel_agents(
            session_id, detection_agents, data
        )

        # Step 3: Aggregate detection results
        all_findings = []
        max_confidence = 0

        for result in detection_results["results"]:
            if "result" in result:
                findings = result["result"].get("findings", [])
                all_findings.extend(findings)
                confidence = result["result"].get("confidence", 0)
                max_confidence = max(max_confidence, confidence)

        # Step 4: Take action based on findings
        action_input = {
            "findings": all_findings,
            "confidence": max_confidence
        }

        action_result = await self.agents["action"].execute(
            action_input,
            {"memory": session.memory, "context": session.context}
        )

        # Step 5: Update memory and metrics
        detection_record = {
            "timestamp": datetime.now().isoformat(),
            "findings": all_findings,
            "confidence": max_confidence,
            "actions": action_result["actions"],
            "category": "sensitive" if max_confidence > 0.5 else "normal"
        }

        self.memory_bank.add_to_short_term(detection_record)

        # Update session
        self.session_service.update_session(session_id, {
            "memory": [detection_record],
            "metrics": {"scans": 1, "detections": len(all_findings)}
        })

        # Record metrics
        duration = (datetime.now() - start_time).total_seconds()
        self.metrics.record_scan(duration)
        if all_findings:
            self.metrics.record_detection(max_confidence > 0.7)

        # Step 6: A2A Communication - notify other agents
        if action_result["blocked"]:
            await self.a2a_protocol.broadcast(
                "action",
                "THREAT_DETECTED",
                {"severity": "high", "findings": all_findings}
            )

        return {
            "session_id": session_id,
            "findings": all_findings,
            "confidence": max_confidence,
            "actions": action_result,
            "processing_time": duration,
            "status": "blocked" if action_result["blocked"] else "allowed"
        }

    async def continuous_monitoring(self, data_stream: List[Dict], 
                                   session_id: str) -> Dict[str, Any]:
        """Long-running monitoring operation with pause/resume"""
        from .sessions import LongRunningOperation

        operation = LongRunningOperation(f"monitor_{session_id}", self.session_service)

        results = []
        for idx, data in enumerate(data_stream):
            # Create checkpoint every 10 items
            if idx % 10 == 0:
                operation.create_checkpoint({
                    "processed": idx,
                    "results": results[-10:] if results else []
                })

            # Check if should pause
            session = self.session_service.get_session(session_id)
            if session.state == SessionState.PAUSED:
                self.logger.info("Monitoring paused, waiting for resume...")
                while session.state == SessionState.PAUSED:
                    await asyncio.sleep(1)
                    session = self.session_service.get_session(session_id)

            # Scan data
            result = await self.scan_data(data, session_id)
            results.append(result)

        return {
            "total_processed": len(data_stream),
            "results": results,
            "checkpoints": len(operation.checkpoints)
        }

    async def evaluate_system(self, test_dataset: List[Dict]) -> Dict[str, Any]:
        """Evaluate the entire system"""
        evaluations = {}

        # Evaluate detection agent
        detection_eval = await self.evaluator.evaluate_detection_accuracy(
            self.agents["detector"],
            test_dataset
        )
        evaluations["detector"] = detection_eval

        # Get overall metrics
        system_metrics = self.metrics.get_metrics()

        # Get evaluation report
        eval_report = self.evaluator.get_evaluation_report()

        return {
            "agent_evaluations": evaluations,
            "system_metrics": system_metrics,
            "evaluation_report": eval_report
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "active_sessions": len(self.session_service.sessions),
            "metrics": self.metrics.get_metrics(),
            "memory_stats": {
                "short_term": len(self.memory_bank.short_term),
                "long_term_categories": len(self.memory_bank.long_term),
                "knowledge_base_patterns": sum(
                    len(v) for v in self.memory_bank.knowledge_base.values()
                )
            },
            "traces": len(self.tracer.traces),
            "a2a_messages": sum(len(v) for v in self.a2a_protocol.message_queue.values())
        }
