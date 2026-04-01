from dotenv import load_dotenv
import os
import anthropic
from mistralai_azure import MistralAzure
from enum import Enum

load_dotenv()


class LLMProvider(Enum):
    CLAUDE = "Claude-Sonnet-4.6"
    MISTRAL = "Mistral-Nemo"


class InferenceAgent:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        azure_api_key = os.getenv("AZURE_AI_API_KEY")
        azure_endpoint = os.getenv("AZURE_AI_ENDPOINT")

        if azure_api_key and azure_endpoint:
            self.mistral_client = MistralAzure(
                azure_api_key=azure_api_key,
                azure_endpoint=azure_endpoint
            )
        else:
            self.mistral_client = None

        self.models = {
            LLMProvider.CLAUDE: "claude-sonnet-4-6",
            LLMProvider.MISTRAL: "azureai"
        }

    def generate_chat_response(self, messages, context=None, provider=LLMProvider.CLAUDE):
        system_message = """
        You are a helpful AI assistant that answers questions based on the provided context.
        Always try to ground your answers in the given context when available.
        If the context doesn't contain relevant information to answer the question,
        clearly state that you cannot find the specific information in the provided context
        and provide a general response based on your knowledge.

        Previous conversations are provided for context. Use them to maintain continuity
        but focus primarily on answering the current question.
        """

        context_str = ""
        if context:
            context_str = "Relevant Document Context:\n"
            for i, doc in enumerate(context, 1):
                context_str += f"\nDocument {i}:\n{doc.page_content}\n"

        chat_history = ""
        if len(messages) > 2:
            chat_history = "Previous Conversation:\n"
            for msg in messages[:-1]:
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
            else:
                return self._prepare_mistral_chat_response(
                    system_message,
                    messages,
                    f"{chat_history}\n{context_str}"
                )
        except Exception as e:
            raise Exception(f"Error during chat inference: {str(e)}")

    def _prepare_claude_chat_response(self, system_message, messages, context_str):
        claude_messages = []

        for i, msg in enumerate(messages):
            if i == 0 and msg["role"] == "user":
                user_content = f"{system_message}\n\n{context_str}\n\nCurrent Question: {msg['content']}"
                claude_messages.append(
                    {"role": "user", "content": user_content})
            else:
                if msg["role"] in ["user", "assistant"]:
                    claude_messages.append(
                        {"role": msg["role"], "content": msg["content"]})

        return {
            "client": self.anthropic_client,
            "model": self.models[LLMProvider.CLAUDE],
            "messages": claude_messages,
            "max_tokens": 8096,
        }

    def _prepare_mistral_chat_response(self, system_message, messages, context_str):
        if not self.mistral_client:
            raise Exception(
                "Mistral Azure client is not configured. Check AZURE_AI_API_KEY and AZURE_AI_ENDPOINT."
            )

        mistral_messages = [{"role": "system", "content": system_message}]

        for i, msg in enumerate(messages):
            if i == 0 and msg["role"] == "user":
                user_content = f"{context_str}\n\nCurrent Question: {msg['content']}"
                mistral_messages.append(
                    {"role": "user", "content": user_content})
            else:
                if msg["role"] in ["user", "assistant"]:
                    mistral_messages.append(
                        {"role": msg["role"], "content": msg["content"]})

        return {
            "client": self.mistral_client,
            "model": self.models[LLMProvider.MISTRAL],
            "messages": mistral_messages,
            "max_tokens": 8000,
            "temperature": 0.7
        }
