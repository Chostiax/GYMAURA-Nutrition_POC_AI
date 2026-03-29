import streamlit as st
from src.data_prep import load_and_prepare_dataset
from src.pipeline import run_pipeline

DATA_PATH = "data/food_item_rows.csv"


@st.cache_data
def get_dataset():
    return load_and_prepare_dataset(DATA_PATH)


st.set_page_config(page_title="GymAura Nutrition PoC AI", layout="wide")

st.title("GymAura Nutrition PoC - AI Version")
st.write("Enter a natural language meal description. The AI parses it into structured food JSON, then the pipeline matches foods and computes nutrition when possible.")

dataset = get_dataset()

model = st.selectbox(
    "Model",
    ["gpt-5-mini", "gpt-5-nano"],
    index=0
)

example_sentences = [
    "I had one banana and a cup of coffee",
    "For lunch I ate grilled chicken with rice",
    "I drank a protein shake after the gym",
    "I didn't eat anything yet today",
    "I had two eggs and a glass of orange juice",
]

selected_example = st.selectbox(
    "Choose an example sentence",
    [""] + example_sentences
)

user_input = st.text_area(
    "Meal description",
    value=selected_example,
    height=120,
    placeholder="Example: I had two eggs, some rice and a bit of chicken"
)

if st.button("Run AI Pipeline"):
    if not user_input.strip():
        st.warning("Please enter a sentence first.")
    else:
        result = run_pipeline(user_input.strip(), dataset, model=model)

        st.subheader("Input")
        st.write(result["input"])

        st.subheader("Detected Items")
        if not result["items"]:
            st.info("No food items detected.")
        else:
            for idx, item in enumerate(result["items"], start=1):
                with st.expander(f"Item {idx}: {item.get('food_text', 'Unknown')}"):
                    st.json(item)

        st.subheader("Meal Totals")
        st.json(result["totals"])

        st.subheader("Raw AI JSON")
        st.code(result["ai_raw_output"], language="json")

        if result.get("ai_usage") is not None:
            st.subheader("Token Usage")
            st.write(result["ai_usage"])