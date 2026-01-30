"""
ScholaRAG Core Module v2.0

Provides the foundational components for the ScholaRAG paper fetching system:
- models: Canonical data models (Paper, etc.)
- adapters: Database-specific API adapters (17 databases planned)
- services: Rate limiting, retry handling, deduplication
- registry: Database configuration and adapter registry
- orchestrator: Central paper fetching coordination

Usage:
    from scripts.core import Paper, fetch_papers, adapter_registry

    # Fetch papers from multiple databases
    results = fetch_papers(
        query="machine learning",
        databases=["semantic_scholar", "openalex", "arxiv"],
        year_start=2020,
        year_end=2024,
    )

    # Use individual adapters
    adapter = adapter_registry.get("semantic_scholar")
    for paper in adapter.search("machine learning"):
        print(paper.title)
"""

__version__ = "2.0.0"

# Models
from .models.paper import Paper

# Services
from .services.rate_limiter import RateLimiter, rate_limiter_registry
from .services.retry_handler import retry_with_backoff, RetrySession
from .services.deduplicator import EnhancedDeduplicator, FileBasedDeduplicator

# Orchestrator
from .orchestrator import PaperFetcherOrchestrator, fetch_papers

__all__ = [
    # Version
    "__version__",
    # Models
    "Paper",
    # Services
    "RateLimiter",
    "rate_limiter_registry",
    "retry_with_backoff",
    "RetrySession",
    "EnhancedDeduplicator",
    "FileBasedDeduplicator",
    # Orchestrator
    "PaperFetcherOrchestrator",
    "fetch_papers",
]
