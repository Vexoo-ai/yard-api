from dotenv import load_dotenv
import os
import anthropic
from mistralai_azure import MistralAzure
from enum import Enum

load_dotenv()

class LLMProvider(Enum):
    CLAUDE = "Claude-Thinking-3.7"
    MISTRAL = "Mistral-Nemo"

class InferenceAgent:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic()
        self.mistral_client = MistralAzure(
            azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
            azure_api_key=os.getenv("AZURE_AI_API_KEY")
        )
        
        # Default models
        self.models = {
            LLMProvider.CLAUDE: "claude-3-7-sonnet-20250219",
            LLMProvider.MISTRAL: "azureai"
        }
    
    def generate_chat_response(self, messages, context=None, provider=LLMProvider.CLAUDE):
        """
        Generate a streaming response in chat format with message history
        """
        system_message = """
        You are a helpful AI assistant that answers questions based on the provided context.
        Always try to ground your answers in the given context when available.
        If the context doesn't contain relevant information to answer the question,
        clearly state that you cannot find the specific information in the provided context
        and provide a general response based on your knowledge.
        
        Previous conversations are provided for context. Use them to maintain continuity
        but focus primarily on answering the current question.
        """

        # Format context if provided
        context_str = ""
        if context:
            context_str = "Relevant Document Context:\n"
            for i, doc in enumerate(context, 1):
                context_str += f"\nDocument {i}:\n{doc.page_content}\n"

        # Format chat history
        chat_history = ""
        if len(messages) > 2:  # If there's history beyond the current exchange
            chat_history = "Previous Conversation:\n"
            # Get all messages except the last user question
            for msg in messages[:-1]:
                # Only include the final answer, not the thinking process
                content = msg["content"]
                if msg["role"] == "assistant" and "<think>" in content:
                    content = content.split("</think>")[-1].strip()
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {content}\n"

        try:
            if provider == LLMProvider.CLAUDE:
                return self._prepare_claude_chat_response(
                    system_message,
                    messages,
                    f"{chat_history}\n{context_str}"
                )
            else:  # MISTRAL
                return self._prepare_mistral_chat_response(
                    system_message,
                    messages,
                    f"{chat_history}\n{context_str}"
                )
        except Exception as e:
            raise Exception(f"Error during chat inference: {str(e)}")
    
    def _prepare_claude_chat_response(self, system_message, messages, context_str):
        """Prepare parameters for Claude chat responses"""
        
        # Format messages for Claude
        claude_messages = []
        
        # Add context and system message to the first user message
        for i, msg in enumerate(messages):
            if i == 0 and msg["role"] == "user":
                user_content = f"{system_message}\n\n{context_str}\n\nCurrent Question: {msg['content']}"
                claude_messages.append({"role": "user", "content": user_content})
            else:
                if msg["role"] in ["user", "assistant"]:
                    # Clean assistant responses - no need to look for <think> tags in Claude's case
                    # as they're part of the event structure, not the content
                    content = msg["content"]
                    claude_messages.append({"role": msg["role"], "content": content})
        
        # Return the parameters needed to create a stream
        return {
            "client": self.anthropic_client,
            "model": self.models[LLMProvider.CLAUDE],
            "messages": claude_messages,
            "max_tokens": 20000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 16000
            }
        }
        