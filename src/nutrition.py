# We only compute item nutrition when we can estimate grams.
# If quantity or portion size is too vague, we keep nutrition as None.

def estimate_grams(food_text: str, quantity: float | None, unit: str | None) -> float | None:
    food_text = food_text.lower().strip()

    if quantity is None:
        return None

    if unit in {"g", "gram", "grams"}:
        return quantity
    if unit in {"kg", "kilogram", "kilograms"}:
        return quantity * 1000
    if unit in {"ml", "milliliter", "milliliters"}:
        return quantity
    if unit in {"l", "liter", "liters"}:
        return quantity * 1000
    
    if quantity is not None and unit is None:
        return quantity * 100.0
    if unit == "g" and quantity is not None:
        return float(quantity)
    
    if unit in {"cup", "cups"}:
        if "rice" in food_text:
            return quantity * 158
        if "pasta" in food_text:
            return quantity * 140
        if "milk" in food_text or "juice" in food_text or "coffee" in food_text or "tea" in food_text:
            return quantity * 240
        if "cereal" in food_text:
            return quantity * 40
        if "yogurt" in food_text:
            return quantity * 245
        if "vegetable" in food_text:
            return quantity * 120


    if unit in {"bowl", "bowls"}:
        if "rice" in food_text:
            return quantity * 180
        if "cereal" in food_text:
            return quantity * 45

    if unit in {"slice", "slices"}:
        if "bread" in food_text:
            return quantity * 30
        if "pizza" in food_text:
            return quantity * 107
        if "cheesecake" in food_text:
            return quantity * 125

    if unit in {"glass", "glasses"}:
        if "juice" in food_text or "milk" in food_text or "tea" in food_text:
            return quantity * 240

    if unit in {"can", "cans"}:
        if any(x in food_text for x in ["coke", "cola", "soda", "soft drink"]):
            return quantity * 355

    if unit in {"bottle", "bottles"}:
        if "tea" in food_text or "juice" in food_text:
            return quantity * 500

    if unit in {"bag", "bags"}:
        if "chips" in food_text:
            return quantity * 28

    if unit in {"scoop", "scoops"}:
        if "salad" in food_text:
            return quantity * 120
        if "protein" in food_text:
            return quantity * 30

    if unit in {"plate", "plates"}:
        if "spaghetti" in food_text or "pasta" in food_text:
            return quantity * 250

    if unit is None:
        if "egg" in food_text:
            return quantity * 50
        if "banana" in food_text:
            return quantity * 118
        if "apple" in food_text:
            return quantity * 182
        if "peach" in food_text:
            return quantity * 150
        if "date" in food_text:
            return quantity * 8
        if "almond" in food_text:
            return quantity * 1.2
        if "strawberry" in food_text:
            return quantity * 12
        if "pizza" in food_text:
            return quantity * 107
        if "muffin" in food_text:
            return quantity * 113
        if "shake" in food_text or "smoothie" in food_text:
            return quantity * 325
        if "wrap" in food_text:
            return quantity * 180
        if "cookie" in food_text:
            return quantity * 16
        if "cappuccino" in food_text or "latte" in food_text or "coffee" in food_text:
            return quantity * 240
        if "potato" in food_text:
            return quantity * 173
        if "cheesecake" in food_text:
            return quantity * 125
        if "coffee" in food_text:
            return quantity *240

    return None


def compute_item_nutrition(dataset_row, grams: float | None) -> dict | None:
    if dataset_row is None or grams is None:
        return None

    factor = grams / 100.0

    def scaled(value):
        try:
            if value is None:
                return 0.0
            return round(float(value) * factor, 2)
        except Exception:
            return 0.0

    return {
        "calories": scaled(dataset_row.get("calories_per_100g")),
        "protein_g": scaled(dataset_row.get("protein_per_100g")),
        "carbs_g": scaled(dataset_row.get("carbs_per_100g")),
        "fat_g": scaled(dataset_row.get("fat_per_100g")),
    }


def sum_nutrition(items: list[dict]) -> dict:
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    for item in items:
        nutrition = item.get("nutrition")
        if not nutrition:
            continue
        totals["calories"] += nutrition.get("calories", 0.0)
        totals["protein_g"] += nutrition.get("protein_g", 0.0)
        totals["carbs_g"] += nutrition.get("carbs_g", 0.0)
        totals["fat_g"] += nutrition.get("fat_g", 0.0)
    for key in totals:
        totals[key] = round(totals[key], 2)
    return totals
