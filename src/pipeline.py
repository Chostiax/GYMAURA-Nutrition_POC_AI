from src.ai_extractor import extract_foods_with_ai
from src.matcher import match_food_to_dataset
from src.nutrition import estimate_grams, compute_item_nutrition


def run_pipeline(text: str, dataset, model: str = "gpt-5-mini") -> dict:
    """
    End-to-end AI pipeline for one input sentence:
    AI parse -> dataset match -> gram estimation -> nutrition totals
    """
    ai_result = extract_foods_with_ai(text, model=model)

    if ai_result["no_food"]:
        return {
            "input": text,
            "items": [],
            "totals": {
                "calories": 0.0,
                "protein_g": 0.0,
                "carbs_g": 0.0,
                "fat_g": 0.0,
            },
            "ai_usage": ai_result["usage"],
            "ai_raw_output": ai_result["raw_output"],
        }

    totals = {
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
    }

    final_items = []

    for extracted in ai_result["items"]:
        food_text = extracted["food_text"]
        quantity = extracted["quantity"]
        unit = extracted["unit"]

        match = match_food_to_dataset(food_text, dataset)

        grams = None
        nutrition = None
        matched_description = None
        match_score = 0.0

        if match["matched"]:
            row_index = match.get("row_index")
            matched_row = dataset.iloc[row_index] if row_index is not None else None

            matched_description = match.get("description")
            match_score = match.get("score", 0.0)

            grams = estimate_grams(food_text, quantity, unit)

            if matched_row is not None and grams is not None:
                nutrition = compute_item_nutrition(matched_row, grams)

                if nutrition:
                    totals["calories"] += nutrition.get("calories", 0.0)
                    totals["protein_g"] += nutrition.get("protein_g", 0.0)
                    totals["carbs_g"] += nutrition.get("carbs_g", 0.0)
                    totals["fat_g"] += nutrition.get("fat_g", 0.0)

        final_items.append({
            "raw_segment": food_text,
            "food_text": food_text,
            "normalized_query": match.get("normalized_query", food_text),
            "quantity": quantity,
            "unit": unit,
            "grams": grams,
            "matched": match["matched"],
            "matched_description": matched_description,
            "match_type": match.get("match_type"),
            "match_score": match_score,
            "needs_clarification": not match["matched"],
            "nutrition": nutrition,
        })

    totals = {k: round(v, 2) for k, v in totals.items()}

    return {
        "input": text,
        "items": final_items,
        "totals": totals,
        "ai_usage": ai_result["usage"],
        "ai_raw_output": ai_result["raw_output"],
    }