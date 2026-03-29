from rapidfuzz import fuzz, process

from src.data_prep import normalize_food_text


ALIASES = {
    "eggs": "egg",
    "fried eggs": "egg",
    "boiled eggs": "egg",
    "scrambled eggs": "egg",
    "white rice": "rice",
    "brown rice": "rice",
    "basmati rice": "rice",
    "jasmine rice": "rice",
    "chicken breast": "chicken",
    "grilled chicken": "chicken",
    "black coffee": "coffee",
    "latte": "coffee",
    "cappuccino": "coffee",
    "oat milk": "oat milk",
    "almonds": "almond",
    "dates": "date",
    "cereals": "cereal",
    "toast": "bread",
    "fries": "fry",
    "orange juice": "orange juice",
    "greek yogurt": "yogurt",
    "yoghurt": "yogurt",
    "berries": "berry",
    "vegetables": "vegetable",
    "coke": "cola",
    "iced tea": "tea",
    "tuna omelet": "omelet",
    "green curry": "curry",
    "chicken caesar wrap": "wrap",
    "spaghetti bolognese": "spaghetti",
    "roasted vegetables": "vegetable",
    "grilled salmon": "salmon",
    "tomato salsa": "salsa",
    "tiramisu": "cheesecake",
    "new york cheesecake": "cheesecake",
    "ribeye": "steak",
    
}


def apply_alias(food_text: str) -> str:
    normalized = normalize_food_text(food_text, drop_generic_words=False)
    return ALIASES.get(normalized, normalized)


def token_overlap_score(query: str, candidate: str) -> float:
    query_tokens = set(query.split())
    candidate_tokens = set(candidate.split())
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(candidate_tokens))
    return overlap / len(query_tokens)

# Matching strategy order:
# 1. alias normalization
# 2. exact clean-text match
# 3. token overlap score
# 4. fuzzy fallback
# This is safer than fuzzy-only matching for generic food words.

def rank_token_candidate(query: str, candidate: str) -> tuple:
    query_tokens = query.split()
    candidate_tokens = candidate.split()
    length_penalty = abs(len(candidate_tokens) - len(query_tokens))
    starts_with_query = 1 if candidate.startswith(query) else 0
    first_token_match = 1 if query_tokens and candidate_tokens and candidate_tokens[0] == query_tokens[0] else 0
    return (starts_with_query, first_token_match, -length_penalty)


def match_food_to_dataset(food_text: str, dataset, fuzzy_threshold: int = 84) -> dict:
    query = apply_alias(food_text)
    query_tokens = set(query.split())

    exact_matches = dataset[dataset["clean_description"] == query]
    if len(exact_matches) > 0:
        row = exact_matches.iloc[0]
        return {
            "matched": True,
            "match_type": "exact",
            "score": 100.0,
            "row_index": int(row.name),
            "description": row["english_description"],
            "normalized_query": query,
        }

    best_idx = None
    best_score = -1.0
    best_rank = (-1, -1, float("-inf"))

    for idx, candidate in dataset["clean_description"].items():
        candidate_tokens = set(candidate.split())
        if query_tokens and query_tokens.issubset(candidate_tokens):
            overlap_score = token_overlap_score(query, candidate)
            rank = rank_token_candidate(query, candidate)
            if overlap_score > best_score or (overlap_score == best_score and rank > best_rank):
                best_score = overlap_score
                best_rank = rank
                best_idx = idx

    if best_idx is not None:
        row = dataset.loc[best_idx]
        return {
            "matched": True,
            "match_type": "token_overlap",
            "score": round(best_score * 100, 1),
            "row_index": int(row.name),
            "description": row["english_description"],
            "normalized_query": query,
        }

    choices = dataset["clean_description"].tolist()
    best_match = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
    if best_match is not None:
        matched_text, score, match_index = best_match
        if score >= fuzzy_threshold:
            row = dataset.iloc[match_index]
            return {
                "matched": True,
                "match_type": "fuzzy",
                "score": round(score, 1),
                "row_index": int(row.name),
                "description": row["english_description"],
                "normalized_query": query,
            }

    return {
        "matched": False,
        "match_type": None,
        "score": 0.0,
        "row_index": None,
        "description": None,
        "normalized_query": query,
    }
