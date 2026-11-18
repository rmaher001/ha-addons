def pytest_configure(config):
    config.addinivalue_line("markers", "requirement(text): the behaviour a test pins")
