import ast
import json
import re
from typing import Any

import pandas as pd


TARGET_NUTRIENTS = {
    "calories": ["energy", "kcal", "calorie", "calories"],
    "protein": ["protein"],
    "carbs": ["carbohydrate", "carbohydrates", "carbs"],
    "fat": ["fat", "total lipid", "lipid"],
}

GENERIC_DESCRIPTION_WORDS = {
    "raw", "cooked", "fresh", "regular", "unenriched", "without", "with",
    "salt", "boiled", "steamed", "grilled", "roasted", "fried",
    "broiled", "baked", "dry", "form", "frozen", "prepared", "plain",
}

# CSV exports are inconsistent: some rows contain valid JSON,
# others look like Python literals. We try both parsers.

def safe_parse_structure(value: Any):
    if pd.isna(value):
        return None
    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return None


def extract_english_description(description_value: Any) -> str:
    parsed = safe_parse_structure(description_value)

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("lang") == "en":
                return str(item.get("description", "")).strip().lower()

    if isinstance(parsed, dict):
        if "en" in parsed:
            return str(parsed["en"]).strip().lower()
        if parsed.get("lang") == "en":
            return str(parsed.get("description", "")).strip().lower()

    return str(description_value).strip().lower() if description_value else ""

# For matching, we normalize descriptions to lowercase and
# reduce simple plurals so food phrases align better.

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split()).strip()


def singularize_simple(word: str) -> str:
    irregular = {
        "eggs": "egg",
        "bananas": "banana",
        "apples": "apple",
        "dates": "date",
        "almonds": "almond",
        "cereals": "cereal",
        "berries": "berry",
        "vegetables": "vegetable",
        "fries": "fry",
        "cookies": "cookie",
        "strawberries": "strawberry",
        "blueberries": "blueberry",
        "cappuccinos": "cappuccino",
        "asparagus": "asparagus",
        "chips": "chips",
    }
    if word in irregular:
        return irregular[word]
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss") and not word.endswith("us"):
        return word[:-1]
    return word


def normalize_food_text(text: str, drop_generic_words: bool = False) -> str:
    text = normalize_text(text)
    tokens = [singularize_simple(tok) for tok in text.split()]
    if drop_generic_words:
        tokens = [tok for tok in tokens if tok not in GENERIC_DESCRIPTION_WORDS]
    return " ".join(tokens).strip()


def iter_nutrient_items(nutrients_value: Any):
    parsed = safe_parse_structure(nutrients_value)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                yield item
    elif isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item

# Nutrition values are stored per 100g in the dataset.
# Later, nutrition.py scales them to the estimated serving size.

def extract_macro_value(nutrients_value: Any, target_macro: str) -> float | None:
    aliases = TARGET_NUTRIENTS[target_macro]
    for item in iter_nutrient_items(nutrients_value):
        nutrient_name = str(item.get("name", "")).lower()
        amount = item.get("amount", item.get("value", None))
        if amount is None:
            continue
        if any(alias in nutrient_name for alias in aliases):
            try:
                return float(amount)
            except Exception:
                return None
    return None


def load_and_prepare_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df["english_description"] = df["description"].apply(extract_english_description)
    df["clean_description"] = df["english_description"].apply(
        lambda x: normalize_food_text(x, drop_generic_words=True)
    )
    df["calories_per_100g"] = df["nutrients"].apply(lambda x: extract_macro_value(x, "calories"))
    df["protein_per_100g"] = df["nutrients"].apply(lambda x: extract_macro_value(x, "protein"))
    df["carbs_per_100g"] = df["nutrients"].apply(lambda x: extract_macro_value(x, "carbs"))
    df["fat_per_100g"] = df["nutrients"].apply(lambda x: extract_macro_value(x, "fat"))
    df = df[df["clean_description"].astype(str).str.len() > 0].reset_index(drop=True)
    return df
