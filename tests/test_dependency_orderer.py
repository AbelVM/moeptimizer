"""Tests for dependency orderer."""


from moeptimizer.dependency_orderer import (
    DependencyOrderer,
    get_dependency_orderer,
)


class TestDependencyOrderer:
    def test_empty_orderer(self) -> None:
        """Empty orderer has no state."""
        orderer = DependencyOrderer()
        assert orderer is not None

    def test_order_by_dependencies(self) -> None:
        """Order messages by dependencies."""
        orderer = DependencyOrderer()
        messages = [
            {"role": "user", "content": "Task 1"},
            {"role": "user", "content": "Task 2"},
        ]
        result = orderer.order_by_dependencies(messages)
        assert len(result) == len(messages)

    def test_group_related(self) -> None:
        """Group related messages."""
        orderer = DependencyOrderer()
        messages = [
            {"role": "user", "content": "import os\nimport sys"},
            {"role": "user", "content": "import json"},
        ]
        result = orderer.group_related(messages)
        assert len(result) == len(messages)

    def test_singleton(self) -> None:
        """Get dependency orderer returns new instance each time."""
        o1 = get_dependency_orderer()
        o2 = get_dependency_orderer()
        # Function returns new instances
        assert isinstance(o1, DependencyOrderer)
        assert isinstance(o2, DependencyOrderer)
