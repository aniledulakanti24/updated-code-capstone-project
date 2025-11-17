import asyncio
from datetime import datetime
from .core import DLPSystem
from .a2a import A2AMessage


async def main_demo():
    dlp_system = DLPSystem()

    # Basic scan
    test_data = {
        "text": "Hi, my credit card is 1234-5678-9012-3456 and SSN is 123-45-6789. This is confidential."
    }

    result = await dlp_system.scan_data(test_data)
    print(f"Session: {result['session_id']}")
    print(f"Findings: {len(result['findings'])}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Status: {result['status'].upper()}")
    print()

    # Parallel agents
    parallel_data = {"text": "Secret API key: sk_live_abc123xyz456"}
    session_id = f"parallel_session_{datetime.now().timestamp()}"
    agents_to_run = [
        dlp_system.agents["detector"],
        dlp_system.agents["llm_analyzer"]
    ]

    parallel_result = await dlp_system.orchestrator.run_parallel_agents(
        session_id, agents_to_run, parallel_data
    )
    print(f"Parallel pipeline results: {parallel_result}")

    # Simple A2A demo
    message = A2AMessage("detector", "action", "HIGH_RISK_DETECTED", {"risk_score": 0.95})
    dlp_system.a2a_protocol.send_message(message)
    received = dlp_system.a2a_protocol.receive_messages("action")
    print(f"Messages received by action: {len(received)}")
