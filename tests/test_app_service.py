import unittest

from fastapi.testclient import TestClient

from src.api import app
from src.app_service import (
    get_demo_day_report,
    get_runtime_status,
    handle_chat_message,
    initial_chat_context,
)


class AppServiceTests(unittest.TestCase):
    def test_chat_service_returns_response(self):
        result = handle_chat_message("похожие на плов")
        self.assertTrue(result["ok"])
        self.assertIn("Похожие рецепты", result["response"])

    def test_runtime_status_contains_core_sections(self):
        status = get_runtime_status()
        self.assertIn("datasets", status)
        self.assertIn("nlp", status)
        self.assertIn("vision", status)
        self.assertTrue(status["datasets"]["recipenlg_ready"])

    def test_repeated_category_query_rotates_recipe(self):
        chat_state = initial_chat_context()
        first = handle_chat_message("ужин", context=chat_state)
        second = handle_chat_message("что на ужин", context=chat_state)
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertNotEqual(first.get("recipe_title"), "")
        self.assertNotEqual(second.get("recipe_title"), "")
        self.assertNotEqual(first["recipe_title"], second["recipe_title"])
        self.assertEqual(first["query_bucket"], second["query_bucket"])

    def test_same_category_bucket_for_pizza_queries(self):
        chat_state = initial_chat_context()
        first = handle_chat_message("пицца", context=chat_state)
        second = handle_chat_message("рецепт пиццы", context=chat_state)
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(first["query_bucket"], second["query_bucket"])

    def test_demo_day_report_contains_core_sections(self):
        report = get_demo_day_report()
        self.assertIn("summary", report)
        self.assertIn("criteria", report)
        self.assertIn("architecture", report)
        self.assertIn("scenarios", report)
        self.assertIn("intelligence", report)
        self.assertIn("score", report["intelligence"])
        self.assertIn("components", report["intelligence"])
        self.assertTrue(report["criteria"])


class ApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_chat_endpoint(self):
        response = self.client.post("/chat", json={"message": "похожие на плов"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        self.assertIn("recipe_title", response.json())
        self.assertIn("query_bucket", response.json())

    def test_demo_report_endpoint(self):
        response = self.client.get("/demo/report")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("summary", payload)
        self.assertIn("criteria", payload)


if __name__ == "__main__":
    unittest.main()
