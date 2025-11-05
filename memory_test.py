from memory_manager import MemoryManager

# Choose any memory type: buffer, window, summary, summary_buffer, token_buffer, kg
memory = MemoryManager(memory_type="window", window_size=3)

# Save chat turns
memory.save_conversation("Hello", "Hi there!")
memory.save_conversation("What is AI?", "AI means Artificial Intelligence.")

# Retrieve
print("Chat History:", memory.get_chat_history())

# Clear if needed
memory.clear_memory()
