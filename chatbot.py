"""
Gemini-powered chatbot module for image captioning system.
Uses Google Generative AI API to answer questions based on image captions.
Supports multilingual translation and conversation history.
"""

from __future__ import annotations

import os
from typing import Optional

import google.generativeai as genai
from deep_translator import GoogleTranslator


class GeminiChatbot:
    """
    A chatbot powered by Google's Gemini API for answering questions
    about generated image captions with multilingual support.
    """

    # Supported languages for translation
    SUPPORTED_LANGUAGES = {
        "English": "en",
        "Hindi": "hi",
        "French": "fr",
        "Spanish": "es",
        "German": "de",
        "Mandarin": "zh-CN",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini chatbot.

        Args:
            api_key: Google API key. If None, will try to load from environment.

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "Google API key not found. "
                "Please set GOOGLE_API_KEY environment variable or pass it directly."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.conversation_history: list[dict] = []
        self.current_caption: Optional[str] = None

    def set_image_context(self, caption: str) -> None:
        """
        Set the current image caption as context for the conversation.

        Args:
            caption: The generated image caption to use as context.
        """
        self.current_caption = caption
        self.conversation_history.clear()

    def _build_system_prompt(self) -> str:
        """Build the system prompt that enforces caption-only responses."""
        return (
            "You are a helpful AI assistant analyzing an image based on its caption. "
            "You must answer questions ONLY based on the provided image description. "
            "Do not make assumptions or add extra details not mentioned in the description. "
            "If the answer cannot be determined from the image description, respond with: "
            "'I cannot determine that from the image description.'"
        )

    def _format_context_message(self, question: str) -> str:
        """
        Format the user question with image context for Gemini.

        Args:
            question: The user's question.

        Returns:
            Formatted message with context.
        """
        return (
            f"Image Description: {self.current_caption}\n\n"
            f"User Question: {question}\n\n"
            f"Instructions: Answer only based on the image description provided above. "
            f"Do not assume extra details. If the answer is not present in the description, "
            f"reply with: 'I cannot determine that from the image description.'"
        )

    def ask(
        self,
        question: str,
        target_language: str = "English",
    ) -> str:
        """
        Ask a question about the current image caption.

        Args:
            question: The user's question.
            target_language: Language for the response (default: English).

        Returns:
            The answer from Gemini.

        Raises:
            ValueError: If no caption has been set or language is not supported.
            Exception: If API call fails.
        """
        if not self.current_caption:
            raise ValueError(
                "No image caption set. Please provide an image caption first."
            )

        if target_language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{target_language}' not supported. "
                f"Supported languages: {', '.join(self.SUPPORTED_LANGUAGES.keys())}"
            )

        try:
            # Build the formatted message with context
            formatted_message = self._format_context_message(question)

            # Get response from Gemini
            response = self.model.generate_content(formatted_message)
            answer = response.text

            # Store in conversation history
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": question,
                    "language": "English",
                }
            )
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "language": target_language,
                }
            )

            # Translate if needed
            if target_language != "English":
                answer = self.translate(answer, "English", target_language)

            return answer

        except Exception as e:
            raise Exception(f"Failed to get response from Gemini API: {str(e)}")

    @staticmethod
    def translate(
        text: str,
        source_language: str = "English",
        target_language: str = "English",
    ) -> str:
        """
        Translate text between supported languages.

        Args:
            text: The text to translate.
            source_language: Source language name.
            target_language: Target language name.

        Returns:
            Translated text.

        Raises:
            ValueError: If language is not supported.
        """
        if source_language == target_language:
            return text

        source_code = GeminiChatbot.SUPPORTED_LANGUAGES.get(source_language)
        target_code = GeminiChatbot.SUPPORTED_LANGUAGES.get(target_language)

        if not source_code or not target_code:
            raise ValueError(
                f"Unsupported language. "
                f"Supported: {', '.join(GeminiChatbot.SUPPORTED_LANGUAGES.keys())}"
            )

        try:
            translator = GoogleTranslator(source_language=source_code, target_language=target_code)
            return translator.translate(text)
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")

    def get_conversation_history(self) -> list[dict]:
        """
        Get the current conversation history.

        Returns:
            List of conversation messages with metadata.
        """
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported languages.

        Returns:
            List of supported language names.
        """
        return list(self.SUPPORTED_LANGUAGES.keys())
