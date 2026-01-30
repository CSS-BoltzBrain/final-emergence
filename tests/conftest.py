"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
import sys
import os

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def empty_config():
    """Fixture providing path to empty.yaml config."""
    return 'configs/empty.yaml'


@pytest.fixture
def surround_config():
    """Fixture providing path to surround.yaml config."""
    return 'configs/surround.yaml'


@pytest.fixture
def sample_shopping_list():
    """Fixture providing a sample shopping list."""
    from product import Product
    return [
        Product(name="Milk", category="Dairy"),
        Product(name="Bread", category="Bakery"),
        Product(name="Exit", category="Exit")
    ]
