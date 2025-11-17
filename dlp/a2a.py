import logging
from datetime import datetime
from typing import Dict, Any, List


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
        agents = ["collector", "detector", "action"]
        for agent in agents:
            if agent != sender:
                msg = A2AMessage(sender, agent, message_type, payload)
                self.send_message(msg)
