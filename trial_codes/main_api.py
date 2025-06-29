import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from transformers import AutoTokenizer
import torch
from huggingface_hub import InferenceClient

# --- Global Data and Model Initialization ---
parts_df = None
similarity_model = None
llm_pipeline = None

# --- Configuration ---
CSV_FILEPATH = "Parts.csv"
SIMILARITY_MODEL_NAME = 'all-MiniLM-L6-v2'
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# --- 1. Data Loading ---
def load_data(filepath=CSV_FILEPATH):
    """Loads the parts data from a CSV file."""
    global parts_df
    try:
        parts_df = pd.read_csv(filepath, sep=';')
        parts_df.replace('', np.nan, inplace=True)
        print(f"Data loaded successfully from {filepath}. Shape: {parts_df.shape}")

        parts_df.dropna(subset=['DESCRIPTION'], inplace=True)
        parts_df['CLEANED_DESCRIPTION'] = parts_df['DESCRIPTION'].apply(preprocess_text_for_similarity)
        print("Preprocessing descriptions for similarity...")

        global similarity_model
        if similarity_model is None:
            initialize_models()

        if similarity_model:
            parts_df['EMBEDDING'] = list(similarity_model.encode(parts_df['CLEANED_DESCRIPTION'].tolist(), show_progress_bar=True))
            print("Embeddings pre-computed for similarity search.")
        else:
            print("Warning: Similarity model not initialized, cannot pre-compute embeddings.")
        return True

    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure the CSV file is in the same directory.")
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

# --- Model Initialization ---
def initialize_models():
    """Initializes the NLP models for similarity and the LLM via Hugging Face Inference API."""
    global similarity_model, llm_pipeline
    print(f"Initializing Sentence Transformer Model ({SIMILARITY_MODEL_NAME})...")
    try:
        similarity_model = SentenceTransformer(
            SIMILARITY_MODEL_NAME,
            device='mps' if torch.backends.mps.is_available() else 'cpu'
        )
        print(f"Sentence Transformer Model initialized on {'MPS' if torch.backends.mps.is_available() else 'CPU'}.")
    except Exception as e:
        print(f"Error initializing Sentence Transformer model: {e}")
        similarity_model = None

    print("Initializing Hugging Face Inference API for Mistral-7B...")
    try:
        from huggingface_hub import InferenceClient
        llm_pipeline = InferenceClient(
            model=MISTRAL_MODEL_NAME,
            token="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"
        )
        test_response = llm_pipeline.text_generation("Test", max_new_tokens=10)
        print(f"Test API response: {test_response}")
        print("Hugging Face Inference API initialized successfully.")
    except Exception as e:
        print(f"Error initializing Inference API: {e}")
        llm_pipeline = None

    return similarity_model is not None and llm_pipeline is not None

# --- Task 1: Chatbot Implementation ---
def generate_context_for_part(part_id):
    """Generates a textual context for a given part ID."""
    if parts_df is None:
        return "Data not loaded. Please check the data file."
    part_info = parts_df[parts_df['ID'].str.lower() == part_id.lower()]
    if part_info.empty:
        return None
    part_data = part_info.iloc[0].dropna().to_dict()
    context = (
        f"Part ID: {part_id}, "
        f"Description: {part_data.get('DESCRIPTION', 'N/A')}, "
        f"Material: {part_data.get('Material', 'N/A')}, "
        f"Rated Current: {part_data.get('Rated Current (A)', 'N/A')}A, "
        f"Rated Voltage: {part_data.get('Rated Voltage (V)', 'N/A')}V, "
        f"Size: {part_data.get('Size', 'N/A')}, "
        f"Operating Temperature: {part_data.get('Operating Temperature-Min (Cel)', 'N/A')}C to {part_data.get('Operating Temperature-Max (Cel)', 'N/A')}C."
    )
    return context.strip()

def retrieve_similar_parts_context(query_text, num_results=3):
    """Retrieves context for similar parts based on a natural language query."""
    if similarity_model is None or 'EMBEDDING' not in parts_df.columns:
        return "Similarity search not available: model or embeddings missing."

    query_embedding = similarity_model.encode([preprocess_text_for_similarity(query_text)])
    similarities = cosine_similarity(query_embedding, np.array(parts_df['EMBEDDING'].tolist()))[0]
    top_indices = similarities.argsort()[::-1]
    context = "Here is some information about potentially similar parts:\n"
    count = 0
    for idx in top_indices:
        if count >= num_results:
            break
        if similarities[idx] < 0.7:
            continue
        part_id = parts_df.iloc[idx]['ID']
        part_description = parts_df.iloc[idx]['DESCRIPTION']
        context += f"Part ID: {part_id}, Description: {part_description}, Similarity Score: {similarities[idx]:.4f}.\n"
        count += 1
    if count == 0:
        return "No highly similar parts found for your query in the database."
    return context

def generate_llm_response(user_query, context=""):
    """Generates a response using Hugging Face Inference API with Mistral-7B."""
    global llm_pipeline
    if llm_pipeline is None:
        return "LLM model not initialized. Please check initialization errors."

    system_prompt = (
        "You are a helpful assistant for a parts catalog. Answer questions accurately based *only* on the provided context. "
        "If the information is not in the context, state that you don't know or cannot find the information."
    )
    full_prompt = f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_query} [/INST]"

    try:
        response = llm_pipeline.text_generation(
            full_prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        print(f"API response: {response}")
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        if not response or len(response) < 10:
            return "No meaningful response generated. Please try rephrasing your query."
        return response
    except Exception as e:
        print(f"LLM error: {str(e)}")
        return f"An error occurred with the LLM: {str(e)}"

def chatbot_interface_llm():
    """Interactive chatbot leveraging an LLM for answering questions."""
    if llm_pipeline is None:
        print("Error: LLM not initialized. Please check model setup and try again.")
        return

    print("\n--- Task 1: LLM Chatbot Interface (with RAG capabilities) ---")
    print("Hello! I'm your Parts Information Chatbot. I can answer questions about parts and find similar ones.")
    print("Examples: 'What is the material of part A1?', 'Tell me about A9', 'Find parts similar to A10'.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        context = ""
        part_id_match = re.search(r'\b([aA]\d+)\b', user_input)
        if part_id_match:
            part_id = part_id_match.group(1).upper()
            part_context = generate_context_for_part(part_id)
            if part_context is None:
                context = f"Part with ID '{part_id}' not found in the database."
            else:
                context = part_context
        elif "similar" in user_input.lower() or "alternative" in user_input.lower():
            context = retrieve_similar_parts_context(user_input)
        else:
            context = ""
        response = generate_llm_response(user_input, context)
        print(f"Chatbot: {response}")

# --- Task 2: Data Extraction and Normalization ---
def extract_and_normalize_data():
    print("\n--- Task 2: Data Extraction and Normalization ---")
    if parts_df is None:
        print("Data not loaded. Cannot perform extraction and normalization.")
        return

    columns_to_extract = [
        'ID', 'DESCRIPTION', 'Rated Current (A)', 'Rated Voltage (V)',
        'Rated Voltage(AC) (V)', 'Rated Voltage(DC) (V)', 'Material',
        'Size', 'Height', 'Length in mm', 'Maximum Power Dissipation',
        'Operating Temperature-Max (Cel)', 'Operating Temperature-Min (Cel)',
        'Pre-arcing time-Min (ms)', 'Rated Breaking Capacity (A)',
        'Joule-integral-Nom (J)'
    ]
    extracted_df = parts_df[columns_to_extract].copy()

    def normalize_current(current_str):
        if pd.isna(current_str): return np.nan
        match = re.search(r'(\d+(\.\d+)?)\s*A', str(current_str))
        if match:
            return float(match.group(1))
        return np.nan

    def normalize_voltage(voltage_str):
        if pd.isna(voltage_str): return np.nan
        match = re.search(r'(\d+)\s*V', str(voltage_str))
        if match:
            return int(match.group(1))
        return np.nan

    def normalize_dimension(dim_str):
        if pd.isna(dim_str): return np.nan
        match = re.search(r'(\d+(\.\d+)?)\s*mm', str(dim_str))
        if match:
            return float(match.group(1))
        return np.nan

    def normalize_power(power_str):
        if pd.isna(power_str): return np.nan
        match = re.search(r'(\d+(\.\d+)?)\s*W', str(power_str))
        if match:
            return float(match.group(1))
        return np.nan

    def normalize_temperature(temp_str):
        if pd.isna(temp_str): return np.nan
        match = re.search(r'([-]?\d+)\s*Cel', str(temp_str))
        if match:
            return int(match.group(1))
        return np.nan

    def normalize_joule_integral(joule_str):
        if pd.isna(joule_str): return np.nan
        match = re.search(r'(\d+(\.\d+)?)\s*J', str(joule_str))
        if match:
            return float(match.group(1))
        return np.nan

    def normalize_time(time_str):
        if pd.isna(time_str): return np.nan
        match = re.search(r'(\d+)\s*ms', str(time_str))
        if match:
            return int(match.group(1))
        return np.nan

    extracted_df['Rated Current (A)_Normalized'] = extracted_df['Rated Current (A)'].apply(normalize_current)
    extracted_df['Rated Voltage (V)_Normalized'] = extracted_df['Rated Voltage (V)'].apply(normalize_voltage)
    extracted_df['Rated Voltage(AC) (V)_Normalized'] = extracted_df['Rated Voltage(AC) (V)'].apply(normalize_voltage)
    extracted_df['Rated Voltage(DC) (V)_Normalized'] = extracted_df['Rated Voltage(DC) (V)'].apply(normalize_voltage)
    extracted_df['Height_Normalized_mm'] = extracted_df['Height'].apply(normalize_dimension)
    extracted_df['Length in mm_Normalized'] = extracted_df['Length in mm'].apply(normalize_dimension)
    extracted_df['Maximum Power Dissipation_Normalized_W'] = extracted_df['Maximum Power Dissipation'].apply(normalize_power)
    extracted_df['Operating Temperature-Max (Cel)_Normalized'] = extracted_df['Operating Temperature-Max (Cel)'].apply(normalize_temperature)
    extracted_df['Operating Temperature-Min (Cel)_Normalized'] = extracted_df['Operating Temperature-Min (Cel)'].apply(normalize_temperature)
    extracted_df['Pre-arcing time-Min (ms)_Normalized'] = extracted_df['Pre-arcing time-Min (ms)'].apply(normalize_time)
    extracted_df['Rated Breaking Capacity (A)_Normalized'] = extracted_df['Rated Breaking Capacity (A)'].apply(normalize_current)
    extracted_df['Joule-integral-Nom (J)_Normalized'] = extracted_df['Joule-integral-Nom (J)'].apply(normalize_joule_integral)

    print("\nOriginal and Normalized Data for selected columns (first 5 rows):")
    display_cols = [
        'ID', 'DESCRIPTION',
        'Rated Current (A)', 'Rated Current (A)_Normalized',
        'Rated Voltage (V)', 'Rated Voltage (V)_Normalized',
        'Height', 'Height_Normalized_mm',
        'Maximum Power Dissipation', 'Maximum Power Dissipation_Normalized_W',
        'Operating Temperature-Max (Cel)', 'Operating Temperature-Max (Cel)_Normalized'
    ]
    print(extracted_df[display_cols].head().to_string())
    print("\nSummary of Normalized Data (Non-null counts and Dtype):")
    print(extracted_df.info())
    return extracted_df

# --- Task 3: Similar Parts Finder ---
def preprocess_text_for_similarity(text):
    """Cleans description text for embedding."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s\-\.\/x]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_similar_parts(num_alternatives=5):
    print("\n--- Task 3: Similar Parts Finder ---")
    if parts_df is None or similarity_model is None or 'EMBEDDING' not in parts_df.columns:
        print("Data, similarity model, or pre-computed embeddings not loaded. Cannot find similar parts.")
        print("Please ensure `load_data()` ran successfully and initialized embeddings.")
        return

    results = []
    description_embeddings = np.array(parts_df['EMBEDDING'].tolist())
    print("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(description_embeddings)
    print("Similarity matrix calculated.")

    for i, (original_idx, row) in enumerate(parts_df.iterrows()):
        original_id = row['ID']
        original_description = row['DESCRIPTION']
        similar_indices_in_matrix = similarity_matrix[i].argsort()[::-1][1:num_alternatives+1]
        alternative_parts = []
        for sim_idx_in_matrix_array in similar_indices_in_matrix:
            alt_row = parts_df.iloc[sim_idx_in_matrix_array]
            alt_id = alt_row['ID']
            alt_description = alt_row['DESCRIPTION']
            similarity_score = similarity_matrix[i, sim_idx_in_matrix_array]
            alternative_parts.append({
                'Alternative_ID': alt_id,
                'Alternative_Description': alt_description,
                'Similarity_Score': similarity_score
            })
        results.append({
            'Original_ID': original_id,
            'Original_Description': original_description,
            'Alternatives': alternative_parts
        })

    print("\n--- Similar Part Finder Results ---")
    for part_data in results[:10]:
        print(f"\nOriginal Part ID: {part_data['Original_ID']}")
        print(f"Description: \"{part_data['Original_Description']}\"")
        print("Found Alternatives:")
        if part_data['Alternatives']:
            for alt in part_data['Alternatives']:
                print(f"  - ID: {alt['Alternative_ID']}, Similarity: {alt['Similarity_Score']:.4f}, Description: \"{alt['Alternative_Description']}\"")
        else:
            print("  No alternatives found (or not enough data).")
    if len(results) > 10:
        print(f"\n... (Showing only first 10 results out of {len(results)}. Run this task again to see more.)")
    return results

# --- Main Execution Flow ---
def main():
    """Main function to orchestrate the execution of all tasks."""
    print("Starting Parts Analysis Application...")
    if not os.path.exists(CSV_FILEPATH):
        print(f"Error: {CSV_FILEPATH} not found. Please create it with 1000 records.")
        return

    if not initialize_models():
        print("Exiting due to model initialization error. Please check dependencies and internet connection.")
        return

    if not load_data():
        print("Exiting due to data loading error.")
        return

    while True:
        print("\nSelect a task to run:")
        print("1. LLM Chatbot Interface (Task 1)")
        print("2. Data Extraction and Normalization (Task 2)")
        print("3. Similar Parts Finder (Task 3)")
        print("0. Exit Application")
        choice = input("Enter your choice (0-3): ").strip()
        if choice == '1':
            chatbot_interface_llm()
        elif choice == '2':
            extract_and_normalize_data()
        elif choice == '3':
            find_similar_parts()
        elif choice == '0':
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 0 and 3.")

if __name__ == "__main__":
    main()