from typing import List, Dict

class ConversationMemory:
    """
    Manages the chat history for the agent.
    """
    def __init__(self, max_turns: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_turn(self, user_query: str, agent_response: str):
        """
        Adds a interaction to the history.
        """
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": agent_response})
        
        # Prune history if it gets too long (keep latest N turns)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history_block(self) -> str:
        """
        Formats history as a string for the prompt.
        """
        if not self.history:
            return ""
            
        block = "PREVIOUS CONVERSATION:\n"
        for turn in self.history:
            role = "User" if turn["role"] == "user" else "Kai"
            block += f"{role}: {turn['content']}\n"
        return block