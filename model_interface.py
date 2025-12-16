"""
Model Interface Module

This module handles interactions with OpenAI's API for chess move prediction.
It provides functions to query different models and extract chess moves from responses.
"""

import os
import time
from typing import Optional, Any, Dict
from openai import OpenAI
from chess_utils import extract_predicted_move, san_to_uci

try:
    from MAD.utils.openai_utils import num_tokens_from_string  # type: ignore
except Exception:  # pragma: no cover
    def num_tokens_from_string(text: str, model_name: str = "") -> int:
        """
        Fallback token estimator when the true tokenizer is unavailable.
        Uses a simple word-count heuristic.
        """
        if not text:
            return 0
        return max(1, len(text.split()))


class ChessModelInterface:
    """
    Interface for interacting with OpenAI models for chess move prediction.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo-instruct",
        *,
        base_url: Optional[str] = None,
        max_completion_tokens: int = 640,
        default_temperature: float = 0.1,
        default_top_p: float = 1.0,
        retry_attempts: int = 1,
    ):
        """
        Initialize the model interface.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If None, will try to get from environment.
            model_name (str): Model name to use
        """
        # Determine if this is an OpenAI model or open-source model
        normalized_name = (model_name or "").lower()
        is_openai_model = "gpt" in normalized_name
        
        # Select appropriate API key and base URL
        if api_key is None:
            if is_openai_model:
                # OpenAI models need OpenAI API key
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"OpenAI model '{model_name}' requires OPENAI_API_KEY. "
                        f"Set OPENAI_API_KEY in your .env file or use an open-source model with ANANNAS_API_KEY."
                    )
            else:
                # Open-source models use Anannas API key
                api_key = os.getenv("ANANNAS_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"Open-source model '{model_name}' requires ANANNAS_API_KEY. "
                        f"Set ANANNAS_API_KEY in your .env file or use an OpenAI model with OPENAI_API_KEY."
                    )

        if base_url is None:
            if is_openai_model:
                # OpenAI models use OpenAI API
                base_url = (
                    os.getenv("OPENAI_BASE_URL")
                    or os.getenv("OPENAI_API_BASE")
                    # Default: None (uses OpenAI's default endpoint)
                )
            else:
                # Open-source models use Anannas API
                base_url = os.getenv("ANANNAS_API_URL", "https://api.anannas.ai/v1")
                print(f"<debug> : Using Anannas API for open-source model '{model_name}'")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        # Debug: print API setup
        if base_url and "anannas" in base_url.lower():
            print(f"<debug> : Using Anannas API for open-source model '{model_name}'")
            print(f"<debug> : base_url={base_url}")
            print(f"<debug> : API key present: {bool(api_key)}")
        elif is_openai_model:
            print(f"<debug> : Using OpenAI API for OpenAI model '{model_name}'")
            print(f"<debug> : base_url={base_url or 'OpenAI default'}")
            print(f"<debug> : API key present: {bool(api_key)}")

        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.base_url = base_url
        # Reasoning models (like qwen3-4b) need more tokens to complete their reasoning
        # Detect if this is a reasoning model and increase max_tokens accordingly
        normalized_name = (model_name or "").lower()
        is_reasoning_model = "qwen" in normalized_name and "qwen3" in normalized_name
        if is_reasoning_model and max_completion_tokens < 512:
            # Reasoning models need at least 512 tokens to complete their thought process
            self.max_completion_tokens = max(512, max_completion_tokens)
        else:
            self.max_completion_tokens = max_completion_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.retry_attempts = max(0, retry_attempts)
        
        # For Anannas API, all models use chat.completions endpoint
        # For OpenAI, detect based on model name
        is_anannas = base_url and "anannas" in base_url.lower()
        if is_anannas:
            # All Anannas models use chat.completions (same as smoke test)
            self.is_chat_model = True
            print(f"<debug> : Detected Anannas API - using chat.completions for all models")
        else:
            # OpenAI models: instruct models use completions, others use chat.completions
            self.is_chat_model = "instruct" not in normalized_name and not normalized_name.startswith("text-")
            print(f"<debug> : Using OpenAI API - is_chat_model={self.is_chat_model}")

        self._last_prompt_info: Optional[Dict[str, Any]] = None
 
    def _call_model_endpoint(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> tuple[Optional[str], Optional[Any], Optional[str]]:
        max_tokens = max_tokens or self.max_completion_tokens
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p

        if self.is_chat_model:
            # Some Anannas models don't support system messages
            # Known models that don't support system messages:
            models_without_system = [
                "google/gemma-3-12b-it:free",
                "google/gemma-3-4b-it:free",
            ]
            is_anannas = self.base_url and "anannas" in self.base_url.lower()
            model_needs_combined = is_anannas and self.model_name in models_without_system
            
            if model_needs_combined:
                # Combine system and user prompts into a single user message
                # Format: system prompt on first line, then blank line, then user prompt
                combined_user_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                messages = [{"role": "user", "content": combined_user_prompt}]
                print(f"<debug> : Using combined prompt (no system message support)")
            else:
                # Try with system message first (most models support it)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                print(f"<debug> : Using system + user messages")
            
            # Debug: show prompt format (first 200 chars)
            if messages[0].get("role") == "user":
                prompt_preview = messages[0]["content"][:200]
            else:
                prompt_preview = f"System: {messages[0]['content'][:100]}... User: {messages[1]['content'][:100]}"
            print(f"<debug> : Prompt preview: {prompt_preview}...")
            
            # Newer OpenAI models (gpt-5-*, o-series) require max_completion_tokens instead of max_tokens
            normalized_name = (self.model_name or "").lower()
            is_newer_model = any(x in normalized_name for x in ["gpt-5", "o1", "o3", "o4"])
            
            # Some models (like gpt-5-nano) only support default temperature (1.0)
            # Detect if this is a model that doesn't support custom temperature
            is_temperature_restricted = "gpt-5-nano" in normalized_name
            
            # Build API kwargs
            api_kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            
            # Only add temperature if the model supports it
            if not is_temperature_restricted:
                api_kwargs["temperature"] = temperature
                api_kwargs["top_p"] = top_p
            
            # Some models like gpt-5-nano might need more tokens or different handling
            # Increase max_completion_tokens for nano models as they might need more space
            if "nano" in normalized_name:
                adjusted_max_tokens = max(max_tokens, 128)  # Ensure at least 128 tokens
            else:
                adjusted_max_tokens = max_tokens
            
            if is_newer_model:
                api_kwargs["max_completion_tokens"] = adjusted_max_tokens
            else:
                api_kwargs["max_tokens"] = adjusted_max_tokens
            
            # Try the request, fallback to combined if system message fails
            try:
                response = self.client.chat.completions.create(**api_kwargs)
            except Exception as e:
                # If system message failed and we haven't tried combined yet, retry
                error_str = str(e)
                error_str_lower = error_str.lower()
                # Check if it's a temperature error - retry without temperature/top_p
                if "temperature" in error_str_lower and "unsupported" in error_str_lower:
                    print(f"<debug> : Temperature not supported, retrying with default parameters")
                    api_kwargs.pop("temperature", None)
                    api_kwargs.pop("top_p", None)
                    response = self.client.chat.completions.create(**api_kwargs)
                elif ("400" in error_str or "Provider returned error" in error_str) and not model_needs_combined and is_anannas:
                    # Fallback: combine system and user prompts
                    print(f"<debug> : System message failed, falling back to combined prompt")
                    combined_user_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                    messages = [{"role": "user", "content": combined_user_prompt}]
                    api_kwargs["messages"] = messages
                    response = self.client.chat.completions.create(**api_kwargs)
                else:
                    raise
            text_output = ""
            finish_reason = None
            if getattr(response, "choices", None):
                first_choice = response.choices[0]
                message = getattr(first_choice, "message", None)
                text_output = (message.content if message else "") or ""
                finish_reason = getattr(first_choice, "finish_reason", None)
                
                # Debug: log finish reason and content length
                if finish_reason:
                    print(f"<debug> : Finish reason: {finish_reason}")
                if not text_output or text_output.strip() == "":
                    print(f"<debug> : Empty response from model. Finish reason: {finish_reason}")
                    # Check for reasoning field (for reasoning models)
                    reasoning = getattr(message, 'reasoning', None)
                    if reasoning:
                        reasoning_text = reasoning if isinstance(reasoning, str) else str(reasoning)
                        # Use reasoning text as the response - extract functions will parse it
                        text_output = reasoning_text
                        print(f"<debug> : Using reasoning text as response (content was empty)")
                    elif finish_reason == "length":
                        print(f"<debug> : Response truncated (length limit). Consider increasing max_completion_tokens.")
                    elif finish_reason == "content_filter":
                        print(f"<debug> : Response filtered by content policy.")
                    else:
                        print(f"<debug> : Warning: Model returned empty content with finish_reason={finish_reason}")
            text_output = text_output.strip()

            prompt_text = ""
            if messages:
                prompt_text = "\n".join(
                    msg.get("content", "") for msg in messages if isinstance(msg, dict)
                )

            self._last_prompt_info = {
                "prompt_text": prompt_text,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "is_chat": True,
                "model": self.model_name,
            }
            return text_output, response, finish_reason

        # For completion API (instruct models), combine system and user prompts
        # Format: system prompt, blank line, then user prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            combined_prompt = user_prompt
        
        print(f"<debug> : Using completions API (instruct model)")
        print(f"<debug> : Combined prompt preview: {combined_prompt[:200]}...")
        
        response = self.client.completions.create(
            model=self.model_name,
            prompt=combined_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        text_output = ""
        finish_reason = None
        if getattr(response, "choices", None):
            first_choice = response.choices[0]
            text_output = (getattr(first_choice, "text", "") or "").strip()
            finish_reason = getattr(first_choice, "finish_reason", None)
        self._last_prompt_info = {
            "prompt_text": combined_prompt,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "is_chat": False,
            "model": self.model_name,
        }
        return text_output, response, finish_reason

    def query_model_for_move(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        api_delay: float = 0.0,
    ) -> Optional[str]:
        if api_delay > 0:
            time.sleep(api_delay)
        try:
            predicted_move_san, _, _ = self._call_model_endpoint(
                system_prompt,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print("Model output:", repr(predicted_move_san))
            return predicted_move_san
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            error_message = str(e)
            if "503" in error_message or "service temporarily unavailable" in error_message.lower():
                wait_seconds = 60
                print(f"<debug> : Encountered 503/service unavailable error. Sleeping {wait_seconds} seconds before retry.")
                time.sleep(wait_seconds)
            return None

    def query_model_for_move_with_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        api_delay: float = 0.0,
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Call OpenAI model with system and user prompts and return predicted move SAN with token info.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            api_delay (float): Delay in seconds before API call (to avoid rate limits)
            
        Returns:
            tuple[Optional[str], Optional[dict]]: (predicted_move_san, token_info)
        """
        if api_delay > 0:
            time.sleep(api_delay)
        try:
            predicted_move_san, response, finish_reason = self._call_model_endpoint(
                system_prompt,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print("Model output:", repr(predicted_move_san))
            token_info = self._build_token_info(
                response,
                model_override=self.model_name,
                finish_reason_override=finish_reason,
                prompt_context=getattr(self, "_last_prompt_info", None),
                response_text=predicted_move_san,
            ) if response is not None else None
            return predicted_move_san, token_info
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            error_message = str(e)
            if "503" in error_message or "service temporarily unavailable" in error_message.lower():
                wait_seconds = 60
                print(f"<debug> : Encountered 503/service unavailable error. Sleeping {wait_seconds} seconds before retry.")
                time.sleep(wait_seconds)
            return None, None

    def get_move_with_extraction(
        self,
        system_prompt: str,
        user_prompt: str,
        current_turn_number: Optional[int] = None,
        is_white_to_move: bool = True,
        use_gpt4: bool = False,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        retry_attempts: Optional[int] = None,
        force_fallback_move: Optional[str] = None,
        api_delay: float = 0.0,
    ) -> tuple[Optional[str], Optional[str], Optional[dict]]:
        """
        Get move from model and extract SAN move from response.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            current_turn_number (Optional[int]): Current turn number
            is_white_to_move (bool): Whether it's white's turn
            use_gpt4 (bool): Whether to use GPT-4 instead of instruct model
            
        Returns:
            tuple[Optional[str], Optional[str], Optional[dict]]: (raw_response, extracted_san_move, token_info)
        """
        max_tokens = max_tokens or self.max_completion_tokens
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        retries = self.retry_attempts if retry_attempts is None else max(0, retry_attempts)

        attempt = 0
        response_text: Optional[str] = None
        token_info: Optional[Dict[str, Any]] = None
        extracted_san: Optional[str] = None
        base_prompt = system_prompt
        retry_instruction = (
            "\n\nIMPORTANT: Respond with the exact format:\n"
            "Move: <your move in SAN>\n"
            "Plan: <comma-separated SAN moves you expect afterwards (optional)>\n"
            "Always provide a move, even if you must guess."
        )

        while attempt <= retries:
            active_system_prompt = base_prompt if attempt == 0 else base_prompt + retry_instruction
            response_text, token_info = self.query_model_for_move_with_tokens(
                active_system_prompt,
                user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                api_delay=api_delay,
            )
            
            # If we got an empty response, don't retry immediately - it might be a model issue
            if not response_text or response_text.strip() == "":
                print(f"<debug> : Empty response on attempt {attempt + 1}/{retries + 1}. Skipping extraction.")
                if attempt < retries:
                    print(f"<debug> : Will retry with modified prompt...")
                else:
                    print(f"<debug> : All retries exhausted. Returning None.")
                attempt += 1
                continue

            if response_text:
                extracted_san = extract_predicted_move(
                    response_text=response_text,
                    current_turn_number=current_turn_number,
                    is_white_to_move=is_white_to_move,
                )
                if extracted_san:
                    return response_text, extracted_san, token_info

            attempt += 1
            # Increase temperature on retries to get more diverse responses
            # This helps avoid repeating the same incorrect move
            temperature = min(0.8, temperature + 0.15)

        # No move could be extracted - return None instead of default fallback
        # Only use force_fallback_move if explicitly provided
        if force_fallback_move:
            fallback_response = f"Move: {force_fallback_move}"
            fallback_token_info = token_info if token_info is not None else self._empty_token_info("fallback")
            return fallback_response, force_fallback_move, fallback_token_info
        
        # Return None to indicate no move could be extracted
        fallback_token_info = token_info if token_info is not None else self._empty_token_info("no_move_extracted")
        return None, None, fallback_token_info

    def get_uci_move(self, system_prompt: str, user_prompt: str, current_fen: str,
                    current_turn_number: Optional[int] = None,
                    is_white_to_move: bool = True,
                    use_gpt4: bool = False) -> Optional[str]:
        """
        Get UCI move from model.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            current_fen (str): Current board FEN
            current_turn_number (Optional[int]): Current turn number
            is_white_to_move (bool): Whether it's white's turn
            use_gpt4 (bool): Whether to use GPT-4
            
        Returns:
            Optional[str]: UCI move string
        """
        san_move = self.get_move_with_extraction(
            system_prompt,
            user_prompt,
            current_turn_number,
            is_white_to_move,
            use_gpt4,
        )
        
        if not san_move:
            return None
        
        return san_to_uci(current_fen, san_move)

    def _empty_token_info(self, finish_reason: str = "fallback") -> Dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": self.model_name,
            "finish_reason": finish_reason,
        }

    def _build_token_info(
        self,
        response: Any,
        *,
        model_override: Optional[str] = None,
        finish_reason_override: Optional[str] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
        response_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage else prompt_tokens + completion_tokens
        if total_tokens == 0 and prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens

        prompt_text = ""
        model_for_estimation = model_override or self.model_name
        if prompt_context:
            prompt_text = prompt_context.get("prompt_text", "") or ""
            model_for_estimation = prompt_context.get("model", model_for_estimation)

        # Estimate prompt tokens if missing
        if (prompt_tokens is None or prompt_tokens == 0) and prompt_text:
            try:
                prompt_tokens = num_tokens_from_string(prompt_text, model_for_estimation)
            except Exception:
                prompt_tokens = num_tokens_from_string(prompt_text)

        # Estimate completion tokens if missing
        if (completion_tokens is None or completion_tokens == 0) and response_text:
            try:
                completion_tokens = num_tokens_from_string(response_text, model_for_estimation)
            except Exception:
                completion_tokens = num_tokens_from_string(response_text)

        if total_tokens in (None, 0):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        finish_reason = finish_reason_override
        if finish_reason is None and getattr(response, "choices", None):
            first_choice = response.choices[0]
            finish_reason = getattr(first_choice, "finish_reason", None)

        return {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
            "model": model_override or self.model_name,
            "finish_reason": finish_reason,
        }


if __name__ == "__main__":
    # Example usage
    print("Model Interface Module - Example Usage")
    print("=" * 40)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in the environment or .env file")
    else:
        print("OpenAI API key found")
        
        # Create model interface
        model_interface = ChessModelInterface(api_key=api_key)
        print(f"Model interface created with model: {model_interface.model_name}")
        
        # Example system and user prompts
        system_prompt = "You are a chess grandmaster. Provide the next move in standard algebraic notation."
        user_prompt = "1. e4 e5 2. Nf3"
        
        print("\nExample prompts:")
        print(f"System: {system_prompt}")
        print(f"User: {user_prompt}")
        print("\nNote: Actual API calls require valid API key and will incur costs.")

