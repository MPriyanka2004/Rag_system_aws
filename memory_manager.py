from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
    ConversationKGMemory,
)


class MemoryManager:
    """
    Reusable class for managing LangChain memory types.
    Can be used with AWS Bedrock or any other LLM backend.
    """

    def __init__(self, memory_type: str = "buffer", window_size: int = 3, token_limit: int = 1000):
        """
        Args:
            memory_type: Choose from ['buffer', 'window', 'summary', 'summary_buffer', 'token_buffer', 'kg']
            window_size: Number of recent turns to keep (for window memory)
            token_limit: Max tokens to retain (for token buffer memory)
        """
        self.memory_type = memory_type.lower()
        self.window_size = window_size
        self.token_limit = token_limit
        self.memory = self._initialize_memory(self.memory_type)

    def _initialize_memory(self, memory_type: str):
        """Initialize the chosen memory type."""
        if memory_type == "buffer":
            return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        elif memory_type == "window":
            return ConversationBufferWindowMemory(k=self.window_size, memory_key="chat_history", return_messages=True)

        elif memory_type == "summary":
            # Normally requires an LLM, but for AWS Bedrock context you can still use it as a placeholder
            return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        elif memory_type == "summary_buffer":
            return ConversationSummaryBufferMemory(memory_key="chat_history", return_messages=True)

        elif memory_type == "token_buffer":
            return ConversationTokenBufferMemory(memory_key="chat_history", return_messages=True)

        elif memory_type == "kg":
            return ConversationKGMemory(memory_key="chat_history")

        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

    def save_conversation(self, user_input: str, bot_output: str):
        """Save a single user–bot exchange to memory."""
        self.memory.save_context({"input": user_input}, {"output": bot_output})

    def get_chat_history(self):
        """Retrieve current conversation history."""
        return self.memory.load_memory_variables({}).get("chat_history", [])

    def clear_memory(self):
        """Clear all stored conversation data."""
        self.memory.clear()
