import asyncio
import re
import logging
from datetime import datetime
from typing import List, Dict, Any
from .tools import Tool
from abc import ABC, abstractmethod


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
        return {
            "findings": [{
                "type": "pii_detected",
                "method": "ml",
                "confidence": 0.85
            }] if len(text) > 50 else {"findings": []}
        }

    async def _contextual_analysis(self, text: str, context: Dict) -> Dict:
        await asyncio.sleep(0.1)
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
            should_continue = await self._evaluate_continuation(context)
            if not should_continue:
                break

            result = await self._monitor_cycle(input_data, context)
            results.append(result)

            context["iteration"] = iteration
            iteration += 1
            await asyncio.sleep(0.1)

        return {
            "total_iterations": iteration,
            "results": results,
            "status": "completed"
        }

    async def _evaluate_continuation(self, context: Dict) -> bool:
        return context.get("active", True)

    async def _monitor_cycle(self, input_data: Dict, context: Dict) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "monitored"
        }
