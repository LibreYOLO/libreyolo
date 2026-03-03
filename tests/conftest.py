def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end (requires GPU and datasets)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast, no external deps)"
    )
