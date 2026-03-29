# Evaluation is intentionally strict:
# a case is only fully successful when expected foods are detected
# and all predicted items are matched to dataset rows.

from src.matcher import apply_alias
from src.pipeline import run_pipeline

def normalize_expected_food(food_name: str) -> str:
    """
    Expected labels in the test set are normalized the same way as predictions.
    """
    return apply_alias(food_name)


def foods_match(expected_food: str, predicted_food: str) -> bool:
    """
    Small helper to allow partial alignment like:
    expected: rice
    predicted: white rice
    """
    return expected_food == predicted_food


def evaluate_pipeline(test_cases: list[dict], dataset) -> dict:
    """
    Compute the PoC KPIs requested by the supervisor:
    - food detection accuracy
    - quantity detection accuracy (simple cases)
    - dataset matching success
    - end-to-end success
    """
    expected_food_total = 0
    correct_food_total = 0

    quantity_case_total = 0
    correct_quantity_total = 0

    extracted_item_total = 0
    matched_item_total = 0

    successful_case_total = 0
    total_cases = len(test_cases)

    details = []

    for case in test_cases:
        text = case["input"]
        expected = case["expected"]

        result = run_pipeline(text, dataset)

        predicted_items = result["items"]
        predicted_foods = [apply_alias(item["food_text"]) for item in predicted_items]
        expected_foods = [normalize_expected_food(item["food"]) for item in expected]

        # Food detection accuracy
        expected_food_total += len(expected_foods)
        case_food_hits = 0
        used_predicted_indexes = set()

        for expected_food in expected_foods:
            for idx, predicted_food in enumerate(predicted_foods):
                if idx in used_predicted_indexes:
                    continue

                if foods_match(expected_food, predicted_food):
                    correct_food_total += 1
                    case_food_hits += 1
                    used_predicted_indexes.add(idx)
                    break

        # Quantity accuracy:
        # only count simple cases where the expected quantity is explicitly given
        for expected_item in expected:
            if expected_item["quantity"] is None:
                continue

            quantity_case_total += 1
            expected_food = normalize_expected_food(expected_item["food"])

            matched_pred = None
            for pred_item in predicted_items:
                pred_food = apply_alias(pred_item["food_text"])
                if foods_match(expected_food, pred_food):
                    matched_pred = pred_item
                    break

            if matched_pred and matched_pred["quantity"] == float(expected_item["quantity"]):
                correct_quantity_total += 1

        # Dataset matching success:
        # among extracted items, how many got matched to dataset rows
        extracted_item_total += len(predicted_items)
        matched_item_total += sum(1 for item in predicted_items if item["matched"])

        # End-to-end success:
        # for this PoC, we mark a case successful if:
        # - all expected foods were detected
        # - and all predicted items matched the dataset
        # This is strict enough to be meaningful but still fair for a PoC.
        all_detected = case_food_hits == len(expected_foods)
        all_matched = all(item["matched"] for item in predicted_items) if predicted_items else len(expected_foods) == 0

        if all_detected and all_matched:
            successful_case_total += 1

        details.append({
            "input": text,
            "expected": expected,
            "result": result,
            "all_detected": all_detected,
            "all_matched": all_matched,
        })

    metrics = {
        "food_detection_accuracy": round(correct_food_total / expected_food_total, 3) if expected_food_total else 0.0,
        "quantity_detection_accuracy": round(correct_quantity_total / quantity_case_total, 3) if quantity_case_total else 0.0,
        "dataset_matching_success": round(matched_item_total / extracted_item_total, 3) if extracted_item_total else 0.0,
        "end_to_end_success": round(successful_case_total / total_cases, 3) if total_cases else 0.0,
        "total_cases": total_cases,
    }

    return {
        "metrics": metrics,
        "details": details,
    }