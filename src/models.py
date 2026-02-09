from dataclasses import dataclass
from typing import List


@dataclass
class Recipe:
    name: str
    ingredients: List[str]
    calories: float = 0.0

    def __str__(self) -> str:
        return f"{self.name} ({', '.join(self.ingredients)})"


@dataclass
class Ingredient:
    name: str
    allergens: List[str]
    risk_score: float = 0.0

    def __str__(self) -> str:
        return f"{self.name} ({', '.join(self.allergens)})"


@dataclass
class Allergen:
    name: str
    sources: List[str]
    severity: float = 0.0

    def __str__(self) -> str:
        return f"{self.name} ({', '.join(self.sources)})"
