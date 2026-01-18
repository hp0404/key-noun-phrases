"""
Pytest configuration and shared fixtures.

To install required spacy models:
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    python -m spacy download ru_core_news_sm
    python -m spacy download uk_core_news_sm
"""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "english: marks tests requiring English model")
    config.addinivalue_line("markers", "german: marks tests requiring German model")
    config.addinivalue_line("markers", "russian: marks tests requiring Russian model")
    config.addinivalue_line(
        "markers", "ukrainian: marks tests requiring Ukrainian model"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests if required models are not installed."""
    import spacy

    # Check which models are available
    available_models = set()
    for model_name in [
        "en_core_web_sm",
        "de_core_news_sm",
        "ru_core_news_sm",
        "uk_core_news_sm",
    ]:
        try:
            spacy.load(model_name)
            available_models.add(model_name)
        except OSError:
            pass

    # Map test class names to required models
    model_map = {
        "English": "en_core_web_sm",
        "German": "de_core_news_sm",
        "Russian": "ru_core_news_sm",
        "Ukrainian": "uk_core_news_sm",
    }

    for item in items:
        # Check if test class name indicates language requirement
        for lang, model in model_map.items():
            if lang in item.nodeid and model not in available_models:
                skip_marker = pytest.mark.skip(reason=f"{model} model not installed")
                item.add_marker(skip_marker)
                break
