import unittest

from src.pipeline import run_text_pipeline


class PipelineTests(unittest.TestCase):
    def test_similarity_fast_path_for_plov(self):
        result = run_text_pipeline("похожие на плов")
        self.assertTrue(result["handled"])
        self.assertEqual(result["stages"]["decision"]["strategy"], "recipenlg_cosine_fuzzy")
        self.assertIn("Похожие рецепты", result["response"])

    def test_breakfast_recommendation_returns_recipe(self):
        result = run_text_pipeline("подбери завтрак с яйцом")
        self.assertTrue(result["handled"])
        self.assertIn("Рецепт:", result["response"])

    def test_generic_salad_query_returns_recipe(self):
        result = run_text_pipeline("салат")
        self.assertTrue(result["handled"])
        self.assertIn("Рецепт:", result["response"])

    def test_dataset_listing_works(self):
        result = run_text_pipeline("покажи датасеты")
        self.assertTrue(result["handled"])
        self.assertIn("Подключенные датасеты", result["response"])


if __name__ == "__main__":
    unittest.main()
