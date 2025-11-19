import logging
from typing import List, Dict, Any
from .sessions import InMemorySessionService
from .memory import MemoryBank
from .observability import MetricsCollector, TracingService
from .agents import Agent


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
        # Ensure a session exists for the pipeline
        if session is None:
            session = self.session_service.create_session(session_id)

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
        # Ensure a session exists for parallel execution
        if session is None:
            session = self.session_service.create_session(session_id)

        tasks = []
        for agent in agents:
            agent_trace = self.tracer.start_trace(f"agent_{agent.name}", trace_id)
            self.metrics.record_agent_call(agent.name)
            tasks.append(agent.execute(
                input_data, 
                {"memory": session.memory, "context": session.context}
            ))

        import asyncio
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
