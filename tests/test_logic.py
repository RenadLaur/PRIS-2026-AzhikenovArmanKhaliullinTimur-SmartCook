import unittest

from src.logic import check_rules


class CheckRulesTests(unittest.TestCase):
    def test_requires_allergy_info(self):
        verdict = check_rules(
            {
                "has_allergy_info": False,
                "calories": 450,
                "ingredients": ["курица", "рис"],
            }
        )
        self.assertIn("Критическая ошибка", verdict)

    def test_rejects_calories_above_max(self):
        verdict = check_rules(
            {
                "has_allergy_info": True,
                "calories": 1000,
                "ingredients": ["курица", "рис"],
            }
        )
        self.assertIn("выше допустимого порога", verdict)

    def test_flags_blacklist_ingredient(self):
        verdict = check_rules(
            {
                "has_allergy_info": True,
                "calories": 450,
                "ingredients": ["курица", "арахис"],
            }
        )
        self.assertIn("запрещенный ингредиент", verdict)

    def test_accepts_valid_recipe(self):
        verdict = check_rules(
            {
                "has_allergy_info": True,
                "calories": 450,
                "ingredients": ["курица", "овощи"],
            }
        )
        self.assertIn("Успех", verdict)


if __name__ == "__main__":
    unittest.main()
