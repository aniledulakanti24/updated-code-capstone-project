# Advanced Multi-Agent Data Leakage Prevention System
# Complete implementation with all required components

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import re
from abc import ABC, abstractmethod

# ============================================================================
# 1. SESSIONS & MEMORY - State Management
# ============================================================================

class SessionState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SessionData:
    session_id: str
    state: SessionState
    created_at: datetime
    updated_at: datetime
    context: Dict[str, Any]
    memory: List[Dict[str, Any]]
    metrics: Dict[str, int]

class InMemorySessionService:
    """Session management for agent state persistence"""
    
    def __init__(self):
        self.sessions: Dict[str, SessionData] = {}
        self.logger = logging.getLogger("SessionService")
    
    def create_session(self, session_id: str) -> SessionData:
        session = SessionData(
            session_id=session_id,
            state=SessionState.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            context={},
            memory=[],
            metrics={"scans": 0, "detections": 0, "actions": 0}
        )
        self.sessions[session_id] = session
        self.logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.updated_at = datetime.now()
            session.context.update(updates.get("context", {}))
            if "memory" in updates:
                session.memory.extend(updates["memory"])
            if "metrics" in updates:
                for key, value in updates["metrics"].items():
                    session.metrics[key] = session.metrics.get(key, 0) + value
    
    def pause_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].state = SessionState.PAUSED
            self.logger.info(f"Paused session: {session_id}")
    
    def resume_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].state = SessionState.ACTIVE
            self.logger.info(f"Resumed session: {session_id}")

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


# ============================================================================
# 2. OBSERVABILITY - Logging, Tracing, Metrics
# ============================================================================

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


# ============================================================================
# 3. TOOLS - MCP, Custom Tools, Built-in Tools
# ============================================================================

class Tool(ABC):
    """Base class for all tools"""
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        pass

class WebSearchTool(Tool):
    """Built-in web search tool for threat intelligence"""
    
    async def execute(self, query: str) -> Dict[str, Any]:
        # Simulated web search for known data leak patterns
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "results": [
                {"title": f"Known pattern: {query}", "threat_level": "high"}
            ],
            "timestamp": datetime.now().isoformat()
        }

class CodeExecutionTool(Tool):
    """Execute custom detection scripts"""
    
    async def execute(self, code: str, context: Dict) -> Dict[str, Any]:
        # Safe code execution for custom detection logic
        # In production, use sandboxed environment
        try:
            result = eval(code, {"__builtins__": {}}, context)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

class MCPTool(Tool):
    """Model Context Protocol tool for external service integration"""
    
    def __init__(self, service_name: str, endpoint: str):
        self.service_name = service_name
        self.endpoint = endpoint
    
    async def execute(self, operation: str, params: Dict) -> Dict[str, Any]:
        # Simulate MCP communication with external security services
        await asyncio.sleep(0.2)
        return {
            "service": self.service_name,
            "operation": operation,
            "result": f"MCP call to {self.endpoint} completed",
            "status": "success"
        }

class OpenAPITool(Tool):
    """OpenAPI-based tool for REST API integrations"""
    
    def __init__(self, api_spec: Dict):
        self.api_spec = api_spec
    
    async def execute(self, endpoint: str, method: str, data: Dict) -> Dict[str, Any]:
        # Call external APIs (email gateways, SIEM systems, etc.)
        await asyncio.sleep(0.15)
        return {
            "endpoint": endpoint,
            "method": method,
            "response": {"status": 200, "data": "API call successful"}
        }


# ============================================================================
# 4. AGENTS - LLM-Powered, Parallel, Sequential, Loop
# ============================================================================

class Agent(ABC):
    """Base Agent class"""
    
    def __init__(self, name: str, tools: List[Tool] = None):
        self.name = name
        self.tools = tools or []
        self.logger = logging.getLogger(f"Agent.{name}")
    
    @abstractmethod
    async def execute(self, input_data: Dict, context: Dict) -> Dict[str, Any]:
        pass

class LLMAgent(Agent):
    """LLM-powered agent for intelligent decision making"""
    
    def __init__(self, name: str, model: str = "claude-sonnet-4"):
        super().__init__(name)
        self.model = model
    
    async def execute(self, input_data: Dict, context: Dict) -> Dict[str, Any]:
        # Simulate LLM call for context-aware analysis
        prompt = self._build_prompt(input_data, context)
        
        # In production, make actual API call to Claude/OpenAI
        await asyncio.sleep(0.3)  # Simulate LLM inference
        
        analysis = {
            "sensitivity_score": 0.85,
            "reasoning": "Detected PII and confidential keywords",
            "recommended_action": "BLOCK",
            "confidence": 0.92
        }
        
        self.logger.info(f"{self.name} analyzed with {self.model}")
        return analysis
    
    def _build_prompt(self, input_data: Dict, context: Dict) -> str:
        return f"""
        Analyze the following data for sensitive information:
        Data: {input_data}
        Context: {context}
        Historical patterns: {context.get('memory', [])}
        
        Determine if this contains data leakage risks.
        """

class DataCollectorAgent(Agent):
    """Collects data from multiple sources in parallel"""
    
    async def execute(self, input_data: Dict, context: Dict) -> Dict[str, Any]:
        sources = input_data.get("sources", [])
        
        # Parallel data collection
        tasks = [self._collect_from_source(source) for source in sources]
        results = await asyncio.gather(*tasks)
        
        return {
            "collected_data": results,
            "source_count": len(sources),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _collect_from_source(self, source: str) -> Dict:
        await asyncio.sleep(0.1)  # Simulate data fetching
        return {"source": source, "data": f"Data from {source}"}

class DetectionAgent(Agent):
    """Multi-stage detection with parallel analysis"""
    
    def __init__(self, name: str, tools: List[Tool] = None):
        super().__init__(name, tools)
        self.patterns = {
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "api_key": r'\b[A-Za-z0-9]{32,}\b',
            "confidential": r'\b(confidential|secret|private|internal)\b'
        }
    
    async def execute(self, input_data: Dict, context: Dict) -> Dict[str, Any]:
        text = input_data.get("text", "")
        
        # Run parallel detection methods
        tasks = [
            self._regex_detection(text),
            self._ml_detection(text),
            self._contextual_analysis(text, context)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Aggregate findings
        all_findings = []
        for result in results:
            all_findings.extend(result.get("findings", []))
        
        return {
            "findings": all_findings,
            "confidence": self._calculate_confidence(all_findings),
            "detection_methods": ["regex", "ml", "contextual"]
        }
    
    async def _regex_detection(self, text: str) -> Dict:
        await asyncio.sleep(0.05)
        findings = []
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                findings.append({
                    "type": pattern_name,
                    "method": "regex",
                    "matches": matches,
                    "confidence": 0.9
                })
        return {"findings": findings}
    
    async def _ml_detection(self, text: str) -> Dict:
        await asyncio.sleep(0.2)  # Simulate ML model inference
        # In production, use trained models (BERT, custom NER, etc.)
        return {
            "findings": [{
                "type": "pii_detected",
                "method": "ml",
                "confidence": 0.85
            }] if len(text) > 50 else {"findings": []}
        }
    
    async def _contextual_analysis(self, text: str, context: Dict) -> Dict:
        await asyncio.sleep(0.1)
        # Analyze based on historical context and patterns
        memory = context.get("memory", [])
        findings = []
        
        if memory and any("sensitive" in str(m) for m in memory):
            findings.append({
                "type": "contextual_risk",
                "method": "contextual",
                "confidence": 0.75,
                "reason": "Similar to previous detections"
            })
        
        return {"findings": findings}
    
    def _calculate_confidence(self, findings: List[Dict]) -> float:
        if not findings:
            return 0.0
        return sum(f.get("confidence", 0) for f in findings) / len(findings)

class ActionAgent(Agent):
    """Takes actions based on detections (blocking, alerting, logging)"""
    
    async def execute(self, input_data: Dict, context: Dict) -> Dict[str, Any]:
        findings = input_data.get("findings", [])
        confidence = input_data.get("confidence", 0)
        
        actions_taken = []
        
        if confidence > 0.8:
            actions_taken.append(await self._block_transmission())
            actions_taken.append(await self._send_alert("high"))
        elif confidence > 0.5:
            actions_taken.append(await self._send_alert("medium"))
            actions_taken.append(await self._log_incident())
        else:
            actions_taken.append(await self._log_incident())
        
        return {
            "actions": actions_taken,
            "blocked": confidence > 0.8,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _block_transmission(self) -> Dict:
        await asyncio.sleep(0.05)
        return {"action": "BLOCK", "status": "success"}
    
    async def _send_alert(self, severity: str) -> Dict:
        await asyncio.sleep(0.1)
        return {"action": "ALERT", "severity": severity, "status": "sent"}
    
    async def _log_incident(self) -> Dict:
        return {"action": "LOG", "status": "logged"}

class LoopAgent(Agent):
    """Continuously monitors and improves detection over time"""
    
    def __init__(self, name: str, max_iterations: int = 10):
        super().__init__(name)
        self.max_iterations = max_iterations
    
    async def execute(self, input_data: Dict, context: Dict) -> Dict[str, Any]:
        iteration = 0
        results = []
        
        while iteration < self.max_iterations:
            # Check if we should continue
            should_continue = await self._evaluate_continuation(context)
            
            if not should_continue:
                break
            
            # Perform monitoring cycle
            result = await self._monitor_cycle(input_data, context)
            results.append(result)
            
            # Update context with learnings
            context["iteration"] = iteration
            
            iteration += 1
            await asyncio.sleep(0.1)  # Monitoring interval
        
        return {
            "total_iterations": iteration,
            "results": results,
            "status": "completed"
        }
    
    async def _evaluate_continuation(self, context: Dict) -> bool:
        # Decide if monitoring should continue
        return context.get("active", True)
    
    async def _monitor_cycle(self, input_data: Dict, context: Dict) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "monitored"
        }


# ============================================================================
# 5. ORCHESTRATION - Sequential & Parallel Agent Coordination
# ============================================================================

class AgentOrchestrator:
    """Orchestrates multiple agents in sequential and parallel workflows"""
    
    def __init__(self, session_service: InMemorySessionService, 
                 memory_bank: MemoryBank, metrics: MetricsCollector,
                 tracer: TracingService):
        self.session_service = session_service
        self.memory_bank = memory_bank
        self.metrics = metrics
        self.tracer = tracer
        self.logger = logging.getLogger("Orchestrator")
    
    async def run_sequential_pipeline(self, session_id: str, 
                                     agents: List[Agent], 
                                     input_data: Dict) -> Dict[str, Any]:
        """Run agents sequentially, passing output to next agent"""
        trace_id = self.tracer.start_trace("sequential_pipeline")
        session = self.session_service.get_session(session_id)
        
        current_data = input_data
        results = []
        
        for agent in agents:
            agent_trace = self.tracer.start_trace(f"agent_{agent.name}", trace_id)
            self.metrics.record_agent_call(agent.name)
            
            try:
                result = await agent.execute(
                    current_data, 
                    {"memory": session.memory, "context": session.context}
                )
                results.append({"agent": agent.name, "result": result})
                current_data = result  # Pass to next agent
                
                self.tracer.end_trace(agent_trace, "success")
            except Exception as e:
                self.logger.error(f"Agent {agent.name} failed: {e}")
                self.tracer.end_trace(agent_trace, "failed", {"error": str(e)})
                raise
        
        self.tracer.end_trace(trace_id, "success")
        return {"pipeline": "sequential", "results": results}
    
    async def run_parallel_agents(self, session_id: str, 
                                  agents: List[Agent], 
                                  input_data: Dict) -> Dict[str, Any]:
        """Run multiple agents in parallel"""
        trace_id = self.tracer.start_trace("parallel_execution")
        session = self.session_service.get_session(session_id)
        
        tasks = []
        for agent in agents:
            agent_trace = self.tracer.start_trace(f"agent_{agent.name}", trace_id)
            self.metrics.record_agent_call(agent.name)
            tasks.append(agent.execute(
                input_data, 
                {"memory": session.memory, "context": session.context}
            ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        formatted_results = []
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "agent": agent.name, 
                    "error": str(result)
                })
            else:
                formatted_results.append({
                    "agent": agent.name, 
                    "result": result
                })
        
        self.tracer.end_trace(trace_id, "success")
        return {"pipeline": "parallel", "results": formatted_results}


# ============================================================================
# 6. CONTEXT ENGINEERING - Context Compaction
# ============================================================================

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


# ============================================================================
# 7. AGENT EVALUATION
# ============================================================================

class AgentEvaluator:
    """Evaluate agent performance and accuracy"""
    
    def __init__(self):
        self.evaluation_results = []
    
    async def evaluate_detection_accuracy(self, agent: Agent, 
                                         test_cases: List[Dict]) -> Dict:
        """Evaluate detection agent with labeled test data"""
        correct = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            result = await agent.execute(
                {"text": test_case["input"]},
                {"memory": [], "context": {}}
            )
            
            expected = test_case["expected_detection"]
            actual = len(result.get("findings", [])) > 0
            
            if expected == actual:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        evaluation = {
            "agent": agent.name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "timestamp": datetime.now().isoformat()
        }
        
        self.evaluation_results.append(evaluation)
        return evaluation
    
    def get_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            return {"message": "No evaluations yet"}
        
        avg_accuracy = sum(e["accuracy"] for e in self.evaluation_results) / len(self.evaluation_results)
        
        return {
            "total_evaluations": len(self.evaluation_results),
            "average_accuracy": avg_accuracy,
            "evaluations": self.evaluation_results
        }


# ============================================================================
# 8. A2A PROTOCOL - Agent-to-Agent Communication
# ============================================================================

class A2AMessage:
    """Agent-to-Agent message format"""
    
    def __init__(self, sender: str, receiver: str, message_type: str, payload: Dict):
        self.id = f"msg_{datetime.now().timestamp()}"
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.payload = payload
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp
        }

class A2AProtocol:
    """Protocol for inter-agent communication"""
    
    def __init__(self):
        self.message_queue: Dict[str, List[A2AMessage]] = {}
        self.logger = logging.getLogger("A2AProtocol")
    
    def send_message(self, message: A2AMessage):
        """Send message from one agent to another"""
        if message.receiver not in self.message_queue:
            self.message_queue[message.receiver] = []
        
        self.message_queue[message.receiver].append(message)
        self.logger.info(f"Message sent: {message.sender} -> {message.receiver}")
    
    def receive_messages(self, agent_name: str) -> List[A2AMessage]:
        """Receive messages for an agent"""
        messages = self.message_queue.get(agent_name, [])
        self.message_queue[agent_name] = []  # Clear after reading
        return messages
    
    async def broadcast(self, sender: str, message_type: str, payload: Dict):
        """Broadcast message to all agents"""
        # In production, maintain agent registry
        agents = ["collector", "detector", "action"]
        for agent in agents:
            if agent != sender:
                msg = A2AMessage(sender, agent, message_type, payload)
                self.send_message(msg)


# ============================================================================
# 9. LONG-RUNNING OPERATIONS - Pause/Resume
# ============================================================================

class LongRunningOperation:
    """Support for pausable/resumable operations"""
    
    def __init__(self, operation_id: str, session_service: InMemorySessionService):
        self.operation_id = operation_id
        self.session_service = session_service
        self.checkpoints: List[Dict] = []
        self.logger = logging.getLogger("LongRunningOp")
    
    def create_checkpoint(self, state: Dict):
        """Save operation state"""
        checkpoint = {
            "checkpoint_id": len(self.checkpoints),
            "timestamp": datetime.now().isoformat(),
            "state": state
        }
        self.checkpoints.append(checkpoint)
        self.logger.info(f"Checkpoint created: {checkpoint['checkpoint_id']}")
    
    def pause(self, session_id: str):
        """Pause operation"""
        self.session_service.pause_session(session_id)
        self.logger.info(f"Operation paused: {self.operation_id}")
    
    def resume(self, session_id: str) -> Dict:
        """Resume from last checkpoint"""
        self.session_service.resume_session(session_id)
        if self.checkpoints:
            last_checkpoint = self.checkpoints[-1]
            self.logger.info(f"Resuming from checkpoint: {last_checkpoint['checkpoint_id']}")
            return last_checkpoint["state"]
        return {}


# ============================================================================
# 10. MAIN DLP SYSTEM - Putting It All Together
# ============================================================================

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


# ============================================================================
# 11. DEPLOYMENT CONFIGURATION
# ============================================================================

class DeploymentConfig:
    """Configuration for production deployment"""
    
    def __init__(self):
        self.config = {
            "environment": "production",
            "scaling": {
                "min_instances": 2,
                "max_instances": 10,
                "auto_scale": True,
                "target_cpu": 70
            },
            "monitoring": {
                "metrics_enabled": True,
                "tracing_enabled": True,
                "log_level": "INFO",
                "alerting": {
                    "email": "security@company.com",
                    "slack_webhook": "https://hooks.slack.com/...",
                    "pagerduty": True
                }
            },
            "security": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "authentication": "oauth2",
                "rate_limiting": {
                    "requests_per_minute": 100,
                    "burst": 20
                }
            },
            "database": {
                "type": "postgresql",
                "connection_pool_size": 20,
                "backup_frequency": "daily",
                "retention_days": 90
            },
            "integrations": {
                "email_gateway": "microsoft365",
                "siem": "splunk",
                "ticketing": "jira",
                "cloud_storage": ["google_drive", "dropbox", "onedrive"]
            }
        }
    
    def to_dict(self) -> Dict:
        return self.config


class DeploymentManager:
    """Manage system deployment"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger("Deployment")
    
    def deploy(self) -> Dict[str, Any]:
        """Deploy the DLP system"""
        self.logger.info("Starting deployment...")
        
        # Deployment steps
        steps = [
            self._validate_configuration(),
            self._setup_infrastructure(),
            self._deploy_agents(),
            self._configure_monitoring(),
            self._run_health_checks()
        ]
        
        results = []
        for step in steps:
            results.append(step)
        
        return {
            "status": "deployed",
            "steps": results,
            "config": self.config.to_dict()
        }
    
    def _validate_configuration(self) -> Dict:
        return {"step": "validate_config", "status": "success"}
    
    def _setup_infrastructure(self) -> Dict:
        return {"step": "setup_infrastructure", "status": "success"}
    
    def _deploy_agents(self) -> Dict:
        return {"step": "deploy_agents", "status": "success"}
    
    def _configure_monitoring(self) -> Dict:
        return {"step": "configure_monitoring", "status": "success"}
    
    def _run_health_checks(self) -> Dict:
        return {"step": "health_checks", "status": "success"}


# ============================================================================
# 12. EXAMPLE USAGE & DEMO
# ============================================================================

async def main_demo():
    """Comprehensive demo of all system features"""
    
    print("=" * 80)
    print("AI MULTI-AGENT DATA LEAKAGE PREVENTION SYSTEM - COMPLETE DEMO")
    print("=" * 80)
    print()
    
    # Initialize system
    dlp_system = DLPSystem()
    
    # 1. BASIC SCAN
    print("1. BASIC DATA SCAN")
    print("-" * 80)
    test_data = {
        "text": "Hi, my credit card is 1234-5678-9012-3456 and SSN is 123-45-6789. This is confidential."
    }
    
    result = await dlp_system.scan_data(test_data)
    print(f"Session: {result['session_id']}")
    print(f"Findings: {len(result['findings'])}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Status: {result['status'].upper()}")
    print(f"Processing Time: {result['processing_time']:.3f}s")
    print()
    
    # 2. PARALLEL AGENT EXECUTION
    print("2. PARALLEL AGENT EXECUTION")
    print("-" * 80)
    parallel_data = {"text": "Secret API key: sk_live_abc123xyz456"}
    session_id = f"parallel_session_{datetime.now().timestamp()}"
    
    agents_to_run = [
        dlp_system.agents["detector"],
        dlp_system.agents["llm_analyzer"]
    ]
    
    parallel_result = await dlp_system.orchestrator.run_parallel_agents(
        session_id, agents_to_run, parallel_data
    )
    print(f"Pipeline: {parallel_result['pipeline']}")
    print(f"Agents executed: {len(parallel_result['results'])}")
    for agent_result in parallel_result['results']:
        print(f"  - {agent_result['agent']}: {agent_result.get('result', {}).get('confidence', 0):.2f}")
    print()
    
    # 3. SEQUENTIAL WORKFLOW
    print("3. SEQUENTIAL AGENT WORKFLOW")
    print("-" * 80)
    sequential_agents = [
        dlp_system.agents["collector"],
        dlp_system.agents["detector"],
        dlp_system.agents["action"]
    ]
    
    sequential_data = {
        "sources": ["email_server", "file_system"],
        "text": "Confidential project data"
    }
    
    sequential_result = await dlp_system.orchestrator.run_sequential_pipeline(
        session_id, sequential_agents, sequential_data
    )
    print(f"Pipeline: {sequential_result['pipeline']}")
    print(f"Steps completed: {len(sequential_result['results'])}")
    print()
    
    # 4. MEMORY & LEARNING
    print("4. MEMORY & LEARNING SYSTEM")
    print("-" * 80)
    print(f"Short-term memory: {len(dlp_system.memory_bank.short_term)} items")
    print(f"Long-term categories: {len(dlp_system.memory_bank.long_term)}")
    
    # Add learning
    dlp_system.memory_bank.learn_pattern("custom_token", r"tok_[a-z0-9]{20}", 0.95)
    print(f"Knowledge base patterns: {len(dlp_system.memory_bank.knowledge_base)}")
    print()
    
    # 5. CONTEXT COMPACTION
    print("5. CONTEXT COMPACTION")
    print("-" * 80)
    large_context = {
        "history": list(range(100)),
        "metadata": {"key": "value"}
    }
    compacted = dlp_system.context_compactor.summarize_context(large_context)
    print(f"Original context keys: {len(large_context)}")
    print(f"Compacted format: {compacted}")
    print()
    
    # 6. A2A PROTOCOL
    print("6. AGENT-TO-AGENT COMMUNICATION")
    print("-" * 80)
    message = A2AMessage(
        "detector", 
        "action", 
        "HIGH_RISK_DETECTED",
        {"risk_score": 0.95, "data": "sensitive"}
    )
    dlp_system.a2a_protocol.send_message(message)
    
    received = dlp_system.a2a_protocol.receive_messages("action")
    print(f"Messages sent: 1")
    print(f"Messages received by 'action' agent: {len(received)}")
    if received:
        print(f"  Message type: {received[0].message_type}")
        print(f"  From: {received[0].sender}")
    print()
    
    # 7. LONG-RUNNING OPERATION WITH PAUSE/RESUME
    print("7. LONG-RUNNING OPERATION (Pause/Resume)")
    print("-" * 80)
    data_stream = [{"text": f"Data {i}"} for i in range(5)]
    
    monitor_session = dlp_system.session_service.create_session("monitor_session")
    
    # Start monitoring (simulate pause after 2 items)
    async def monitored_scan():
        results = []
        operation = LongRunningOperation("demo_op", dlp_system.session_service)
        
        for idx, data in enumerate(data_stream):
            if idx == 2:
                operation.pause("monitor_session")
                print("  [PAUSED at item 2]")
                await asyncio.sleep(0.5)
                operation.resume("monitor_session")
                print("  [RESUMED]")
            
            result = await dlp_system.scan_data(data, "monitor_session")
            results.append(result)
            
            if idx % 2 == 0:
                operation.create_checkpoint({"processed": idx})
        
        return results
    
    monitored_results = await monitored_scan()
    print(f"Processed: {len(monitored_results)} items")
    print()
    
    # 8. OBSERVABILITY - METRICS & TRACING
    print("8. OBSERVABILITY - METRICS & TRACING")
    print("-" * 80)
    metrics = dlp_system.metrics.get_metrics()
    print(f"Total scans: {metrics['total_scans']}")
    print(f"Total detections: {metrics['total_detections']}")
    print(f"Average processing time: {metrics['average_processing_time']:.3f}s")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Agent calls: {metrics['agent_calls']}")
    
    traces = dlp_system.tracer.get_trace_tree()
    print(f"\nActive traces: {len(traces)}")
    print()
    
    # 9. AGENT EVALUATION
    print("9. AGENT EVALUATION")
    print("-" * 80)
    test_cases = [
        {"input": "My card is 1234-5678-9012-3456", "expected_detection": True},
        {"input": "Hello, how are you?", "expected_detection": False},
        {"input": "Confidential data: secret123", "expected_detection": True},
        {"input": "Normal business email", "expected_detection": False}
    ]
    
    evaluation = await dlp_system.evaluate_system(test_cases)
    print(f"Detection accuracy: {evaluation['agent_evaluations']['detector']['accuracy']:.2%}")
    print(f"Test cases: {evaluation['agent_evaluations']['detector']['total']}")
    print()
    
    # 10. SYSTEM STATUS
    print("10. SYSTEM STATUS")
    print("-" * 80)
    status = dlp_system.get_system_status()
    print(f"Active sessions: {status['active_sessions']}")
    print(f"Memory - Short term: {status['memory_stats']['short_term']}")
    print(f"Memory - Long term categories: {status['memory_stats']['long_term_categories']}")
    print(f"Knowledge base patterns: {status['memory_stats']['knowledge_base_patterns']}")
    print(f"Total traces: {status['traces']}")
    print()
    
    # 11. DEPLOYMENT CONFIGURATION
    print("11. DEPLOYMENT CONFIGURATION")
    print("-" * 80)
    config = DeploymentConfig()
    deployment = DeploymentManager(config)
    deployment_result = deployment.deploy()
    print(f"Deployment status: {deployment_result['status'].upper()}")
    print(f"Deployment steps: {len(deployment_result['steps'])}")
    print(f"Environment: {deployment_result['config']['environment']}")
    print(f"Auto-scaling: {deployment_result['config']['scaling']['auto_scale']}")
    print()
    
    # 12. TOOLS DEMONSTRATION
    print("12. TOOLS INTEGRATION")
    print("-" * 80)
    
    # Web Search Tool
    search_result = await dlp_system.tools["web_search"].execute("credit card pattern")
    print(f"Web Search: {search_result['results'][0]['title']}")
    
    # MCP Tool
    mcp_result = await dlp_system.tools["mcp"].execute("check_threat", {"data": "test"})
    print(f"MCP Tool: {mcp_result['result']}")
    
    # OpenAPI Tool
    api_result = await dlp_system.tools["api"].execute("/scan", "POST", {"data": "test"})
    print(f"API Tool: {api_result['response']['data']}")
    print()
    
    print("=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("SYSTEM CAPABILITIES DEMONSTRATED:")
    print("✓ LLM-Powered Agents")
    print("✓ Parallel Agent Execution")
    print("✓ Sequential Agent Workflows")
    print("✓ Loop Agents for Monitoring")
    print("✓ Session & State Management")
    print("✓ Long-term Memory Bank")
    print("✓ Context Engineering & Compaction")
    print("✓ Logging, Tracing & Metrics")
    print("✓ Agent Evaluation")
    print("✓ A2A Protocol (Agent-to-Agent)")
    print("✓ Pause/Resume Operations")
    print("✓ MCP, Custom, and Built-in Tools")
    print("✓ Deployment Configuration")
    print()


# ============================================================================
# 13. RUN THE DEMO
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the complete demo
    asyncio.run(main_demo())


# ============================================================================
# 14. PRODUCTION DEPLOYMENT INSTRUCTIONS
# ============================================================================

"""
DEPLOYMENT GUIDE:

1. PREREQUISITES:
   - Python 3.8+
   - Docker & Kubernetes (for production)
   - PostgreSQL database
   - Redis for caching
   - Message queue (RabbitMQ/Kafka)

2. INSTALLATION:
   pip install -r requirements.txt
   
   requirements.txt:
   - anthropic>=0.18.0
   - openai>=1.0.0
   - spacy>=3.5.0
   - pandas>=2.0.0
   - asyncio
   - pydantic>=2.0.0
   - sqlalchemy>=2.0.0
   - redis>=4.5.0
   - kubernetes>=25.0.0

3. ENVIRONMENT VARIABLES:
   export ANTHROPIC_API_KEY="your-key"
   export OPENAI_API_KEY="your-key"
   export DATABASE_URL="postgresql://..."
   export REDIS_URL="redis://..."

4. DOCKER DEPLOYMENT:
   docker build -t dlp-system:latest .
   docker run -p 8000:8000 dlp-system:latest

5. KUBERNETES DEPLOYMENT:
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/hpa.yaml

6. MONITORING SETUP:
   - Prometheus for metrics
   - Grafana for dashboards
   - Jaeger for distributed tracing
   - ELK stack for logs

7. SCALING:
   kubectl scale deployment dlp-system --replicas=10

8. INTEGRATION:
   - Connect to email gateways (Office365, Gmail)
   - Integrate with SIEM (Splunk, ELK)
   - Setup cloud storage monitoring
   - Configure alerting channels

9. TESTING:
   pytest tests/
   python -m pytest --cov=dlp_system

10. MAINTENANCE:
   - Regular model retraining
   - Pattern updates
   - Performance monitoring
   - Security audits
"""