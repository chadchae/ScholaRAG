"""
ScholaRAG LLM Provider Abstraction
==================================

Provides a unified interface for multiple LLM providers.
Enables switching between Groq (100x cheaper), Claude, and Ollama.

Cost Comparison (per 100 papers screening):
- Groq Llama 3.3 70B: ~$0.01
- Claude Haiku 4-5: ~$0.15
- Claude Sonnet 3.5: ~$0.45
- Ollama (local): $0

Usage:
    from core.llm_provider import get_llm_provider, LLMConfig

    # Use Groq (default for screening - 100x cheaper)
    provider = get_llm_provider("groq", LLMConfig(model="llama-3.3-70b-versatile"))
    response = provider.complete(prompt, max_tokens=500)

    # Use Claude (high-quality, for complex tasks)
    provider = get_llm_provider("claude", LLMConfig(model="claude-haiku-4-5"))
    response = provider.complete(prompt, max_tokens=500)

    # Use Ollama (completely free, local)
    provider = get_llm_provider("ollama", LLMConfig(model="llama3.2:70b"))
    response = provider.complete(prompt, max_tokens=500)
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    model: str = "llama-3.3-70b-versatile"  # Default to Groq's best model
    temperature: float = 0.1  # Low for deterministic screening
    max_tokens: int = 500
    timeout: int = 30
    max_retries: int = 3
    base_delay: float = 0.5

    # Provider-specific settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    # Cost tracking
    track_tokens: bool = True


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""
    content: str
    model: str
    provider: str

    # Token usage (for cost tracking)
    input_tokens: int = 0
    output_tokens: int = 0

    # Cache info (for Claude)
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    # Metadata
    raw_response: Optional[Any] = None
    latency_ms: float = 0.0

    def get_total_tokens(self) -> int:
        """Get total tokens used"""
        return self.input_tokens + self.output_tokens

    def estimate_cost(self) -> float:
        """
        Estimate cost based on provider and model.

        Pricing (per 1M tokens):
        - Groq Llama 3.3 70B: $0.10 input, $0.29 output
        - Groq Qwen3 32B: $0.10 input, $0.20 output
        - Claude Haiku 4-5: $1.00 input, $5.00 output
        - Claude Sonnet 3.5: $3.00 input, $15.00 output
        - Ollama: $0 (local)
        """
        pricing = {
            "groq": {
                "llama-3.3-70b-versatile": (0.10, 0.29),
                "qwen-qwq-32b": (0.10, 0.20),
                "mixtral-8x7b-32768": (0.10, 0.10),
                "default": (0.10, 0.29),
            },
            "claude": {
                "claude-haiku-4-5": (1.00, 5.00),
                "claude-sonnet-3-5": (3.00, 15.00),
                "claude-opus-4-5": (15.00, 75.00),
                "default": (1.00, 5.00),
            },
            "ollama": {
                "default": (0.0, 0.0),  # Free!
            },
        }

        provider_pricing = pricing.get(self.provider, {"default": (0.0, 0.0)})
        input_price, output_price = provider_pricing.get(
            self.model, provider_pricing["default"]
        )

        cost = (
            (self.input_tokens / 1_000_000) * input_price +
            (self.output_tokens / 1_000_000) * output_price
        )

        return cost


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            max_tokens: Override config max_tokens
            temperature: Override config temperature

        Returns:
            LLMResponse with content and metadata
        """
        pass

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry"""
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise e

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get token usage and cost summary"""
        return {
            "provider": self.__class__.__name__,
            "model": self.config.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
        }


class GroqProvider(LLMProvider):
    """
    Groq LLM Provider - Ultra-fast, ultra-cheap inference.

    Cost: ~$0.01 per 100 papers (100x cheaper than Claude)
    Speed: 500+ tokens/second

    Best models:
    - llama-3.3-70b-versatile: Best quality, $0.10/$0.29 per MTok
    - qwen-qwq-32b: Good quality, $0.10/$0.20 per MTok
    - mixtral-8x7b-32768: Fast, $0.10/$0.10 per MTok
    """

    PROVIDER_NAME = "groq"

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Get API key
        self.api_key = config.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in .env or pass via config.\n"
                "Get your free API key at: https://console.groq.com/keys"
            )

        # Initialize Groq client
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Groq package not installed. Run: pip install groq"
            )

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Generate completion using Groq"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        def _call_api():
            return self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
            )

        response = self._retry_with_backoff(_call_api)

        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        content = response.choices[0].message.content
        usage = response.usage

        llm_response = LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.PROVIDER_NAME,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            raw_response=response,
            latency_ms=latency_ms,
        )

        # Track usage
        if self.config.track_tokens:
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens
            self.total_cost += llm_response.estimate_cost()

        return llm_response


class ClaudeProvider(LLMProvider):
    """
    Claude LLM Provider - High quality with prompt caching.

    Cost: ~$0.15 per 100 papers (with caching)
    Quality: Best for complex reasoning

    Best models:
    - claude-haiku-4-5: Fast, cheap, good for screening
    - claude-sonnet-3-5: Best quality/cost ratio
    - claude-opus-4-5: Best quality, expensive
    """

    PROVIDER_NAME = "claude"

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Get API key
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in .env or pass via config."
            )

        # Initialize Anthropic client
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Run: pip install anthropic"
            )

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Generate completion using Claude with optional caching"""

        start_time = time.time()

        # Build system prompt with caching
        system = None
        if system_prompt:
            if use_cache:
                system = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                system = system_prompt

        def _call_api():
            return self.client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens or self.config.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )

        response = self._retry_with_backoff(_call_api)

        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        content = response.content[0].text
        usage = response.usage

        # Get cache tokens
        cache_creation = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0

        llm_response = LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.PROVIDER_NAME,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            raw_response=response,
            latency_ms=latency_ms,
        )

        # Track usage
        if self.config.track_tokens:
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens
            self.total_cost += llm_response.estimate_cost()

        return llm_response


class OllamaProvider(LLMProvider):
    """
    Ollama LLM Provider - Completely free, runs locally.

    Cost: $0 (local inference)
    Requirement: Ollama installed and running

    Best models:
    - llama3.2:70b: Best quality
    - llama3.2:8b: Fast
    - qwen2.5:32b: Good for coding
    """

    PROVIDER_NAME = "ollama"

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Ollama API base
        self.api_base = config.api_base or os.getenv(
            "OLLAMA_API_BASE", "http://localhost:11434"
        )

        # Test connection
        try:
            import requests
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama not responding")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.api_base}. "
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {e}"
            )

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Generate completion using Ollama"""
        import requests

        start_time = time.time()

        # Build request
        data = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
            }
        }

        if system_prompt:
            data["system"] = system_prompt

        def _call_api():
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=data,
                timeout=self.config.timeout * 10,  # Ollama can be slow
            )
            response.raise_for_status()
            return response.json()

        response = self._retry_with_backoff(_call_api)

        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        content = response.get("response", "")

        # Ollama provides token counts
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        llm_response = LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.PROVIDER_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=response,
            latency_ms=latency_ms,
        )

        # Track usage
        if self.config.track_tokens:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            # Ollama is free!

        return llm_response


# Provider registry
_PROVIDERS: Dict[str, type] = {
    "groq": GroqProvider,
    "claude": ClaudeProvider,
    "ollama": OllamaProvider,
}


def get_llm_provider(
    provider_name: str = "groq",
    config: Optional[LLMConfig] = None,
) -> LLMProvider:
    """
    Get an LLM provider instance.

    Args:
        provider_name: One of "groq", "claude", "ollama"
        config: LLM configuration (uses defaults if not provided)

    Returns:
        LLMProvider instance

    Example:
        # Use Groq (recommended for screening - 100x cheaper)
        provider = get_llm_provider("groq")
        response = provider.complete("Score this paper...")

        # Use Claude (for high-quality tasks)
        provider = get_llm_provider("claude", LLMConfig(model="claude-sonnet-3-5"))

        # Use Ollama (free, local)
        provider = get_llm_provider("ollama", LLMConfig(model="llama3.2:8b"))
    """
    provider_name = provider_name.lower()

    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(_PROVIDERS.keys())}"
        )

    if config is None:
        # Set default model per provider
        default_models = {
            "groq": "llama-3.3-70b-versatile",
            "claude": "claude-haiku-4-5",
            "ollama": "llama3.2:8b",
        }
        config = LLMConfig(model=default_models[provider_name])

    return _PROVIDERS[provider_name](config)


def get_best_provider_for_task(
    task: str = "screening",
    prefer_local: bool = False,
) -> LLMProvider:
    """
    Get the best provider for a specific task.

    Args:
        task: One of "screening", "analysis", "generation"
        prefer_local: If True, prefer Ollama

    Returns:
        LLMProvider instance optimized for the task
    """
    if prefer_local:
        try:
            return get_llm_provider("ollama", LLMConfig(model="llama3.2:8b"))
        except ConnectionError:
            pass  # Fall through to cloud providers

    # Task-optimized provider selection
    task_providers = {
        "screening": ("groq", "llama-3.3-70b-versatile"),  # Cheap, fast
        "analysis": ("groq", "llama-3.3-70b-versatile"),   # Good quality, cheap
        "generation": ("claude", "claude-sonnet-3-5"),     # High quality
    }

    provider_name, model = task_providers.get(
        task, ("groq", "llama-3.3-70b-versatile")
    )

    return get_llm_provider(provider_name, LLMConfig(model=model))
