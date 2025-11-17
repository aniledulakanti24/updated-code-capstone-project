from datetime import datetime
from typing import List, Dict, Any
from .agents import Agent


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
