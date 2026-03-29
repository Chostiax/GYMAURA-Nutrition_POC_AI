from dotenv import load_dotenv
load_dotenv()

import json
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You extract foods and quantities from meal descriptions and return strict JSON.

Return only valid JSON.

Rules:
- Return only foods or drinks actually consumed.
- Ignore context such as time of day, feelings, who prepared the meal, or restaurant mentions.
- Use singular food names when appropriate.
- Extract explicit quantities whenever clearly stated.
- Interpret "a" or "an" before a food as quantity = 1.
- Preserve simple units such as: g, glass, cup, bowl, plate, slice, can, bottle, bag, piece.
- If the sentence says "some <food>", interpret it as quantity = 100 and unit = "g".
- If the sentence says "a bit of <food>", interpret it as quantity = 50 and unit = "g".
- If the sentence says "half a <unit> of <food>", use quantity = 0.5 and keep the unit.
- If a food is a known composite dish (for example: caesar salad, burger, sandwich, pasta dish, pizza, taco, wrap, smoothie), you may decompose it into its main ingredients.
- When decomposing a composite dish, return the inferred ingredients instead of the dish name itself.
- For inferred ingredients, assign reasonable approximate quantities, preferably in grams.
- Prefer a small number of main ingredients rather than too many minor details.
- Do not invent unrealistic ingredients.
- If quantity is not stated and none of the above rules apply, return quantity = null and unit = null.
- If the user says they did not eat anything, set no_food = true and return an empty items list.
- Do not add commentary.

Examples:
Input: "I ate 3 bananas"
Output: {"no_food": false, "items": [{"food_text": "banana", "quantity": 3, "unit": null}]}

Input: "I had some plain yogurt"
Output: {"no_food": false, "items": [{"food_text": "plain yogurt", "quantity": 100, "unit": "g"}]}

Input: "I had two eggs and a glass of orange juice"
Output: {"no_food": false, "items": [{"food_text": "egg", "quantity": 2, "unit": null}, {"food_text": "orange juice", "quantity": 1, "unit": "glass"}]}

Input: "I ate a big caesar salad"
Output: {"no_food": false, "items": [{"food_text": "lettuce", "quantity": 200, "unit": "g"}, {"food_text": "chicken", "quantity": 150, "unit": "g"}, {"food_text": "croutons", "quantity": 50, "unit": "g"}, {"food_text": "parmesan", "quantity": 30, "unit": "g"}, {"food_text": "caesar dressing", "quantity": 70, "unit": "g"}]}

Input: "I didn't eat anything yet today"
Output: {"no_food": true, "items": []}
"""
MEAL_SCHEMA = {
    "name": "meal_parse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "no_food": {"type": "boolean"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "food_text": {"type": "string"},
                        "quantity": {"type": ["number", "null"]},
                        "unit": {"type": ["string", "null"]}
                    },
                    "required": ["food_text", "quantity", "unit"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["no_food", "items"],
        "additionalProperties": False
    }
}


def extract_foods_with_ai(text: str, model: str = "gpt-5-mini") -> dict:
    """
    Use OpenAI to parse a natural-language meal description into structured JSON.
    """
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": MEAL_SCHEMA["name"],
                "strict": True,
                "schema": MEAL_SCHEMA["schema"],
            }
        },
    )

    parsed = json.loads(response.output_text)

    return {
        "no_food": parsed["no_food"],
        "items": parsed["items"],
        "usage": getattr(response, "usage", None),
        "raw_output": response.output_text,
        "response_id": getattr(response, "id", None),
    }