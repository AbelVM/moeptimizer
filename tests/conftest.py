"""Pytest configuration for moeptimizer tests."""


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run e2e tests against real Lemonade server (default: dry run with mocks)",
    )
    parser.addoption(
        "--rounds",
        action="store",
        type=int,
        default=1,
        help="Number of benchmark rounds for live e2e tests",
    )
