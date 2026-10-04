from typing import Optional
from agent.env import Environment
from mock_app.seed import get_all_bills

class MockAppEnvironment(Environment):
    """
    Mock application environment connecting directly to the SQLite database.
    Provides ground truth records without routing through browser interactions.
    """
    name: str = "mock_app"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def ground_truth(self) -> list[dict]:
        """Directly queries SQLite to return all current bill records."""
        return get_all_bills(self.db_path)
