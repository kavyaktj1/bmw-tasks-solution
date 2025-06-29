'''
# Updated environment setup instructions
conda deactivate
conda env remove -n bmw_chatbot
conda create -n bmw_chatbot python=3.11
conda activate bmw_chatbot
conda install pandas=2.3.0 numpy=1.26.4 scikit-learn=1.5.1 sentence-transformers
conda config --add channels conda-forge
pip install pandas==2.3.0 numpy==1.26.4 scikit-learn==1.5.1 sentence-transformers==3.0.1 llama-cpp-python==0.2.85
cd /Users/kavyakt/Downloads/BMW_GenAI_Chatbot
python main.py
'''

import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os  # For checking file existence

# --- LLM Integration specific imports ---
# Use llama-cpp-python for local GGUF model
from llama_cpp import Llama

# --- Global Data and Model Initialization ---
parts_df = None
similarity_model = None  # For sentence embeddings, crucial for RAG
llm_model = None  # For GGUF Mistral model

# --- Configuration ---
CSV_FILEPATH = "Parts.csv"
SIMILARITY_MODEL_NAME = 'all-MiniLM-L6-v2'
MISTRAL_MODEL_PATH = "/Users/kavyakt/Downloads/BMW_GenAI_Chatbot/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# --- 1. Data Loading ---
def load_data(filepath=CSV_FILEPATH):
    """Loads the parts data from a CSV file."""
    global parts_df
    try:
        parts_df = pd.read_csv(filepath, sep=';')
        parts_df.replace('', np.nan, inplace=True)  # Replace empty strings with NaN
        print(f"Data loaded successfully from {filepath}. Shape: {parts_df.shape}")

        # Precompute cleaned descriptions and embeddings for faster similarity search
        parts_df.dropna(subset=['DESCRIPTION'], inplace=True)
        parts_df['CLEANED_DESCRIPTION'] = parts_df['DESCRIPTION'].apply(preprocess_text_for_similarity)
        print("Preprocessing descriptions for similarity...")

        global similarity_model  # Ensure model is loaded before embedding

        if similarity_model is None:
            initialize_models()  # Initialize if not already

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
    """Initializes the NLP models for similarity and the LLM."""
    global similarity_model, llm_model
    print(f"Initializing Sentence Transformer Model ({SIMILARITY_MODEL_NAME})...")
    try:
        similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME)
        print("Sentence Transformer Model initialized.")
    except Exception as e:
        print(f"Error initializing Sentence Transformer model: {e}")
        print("Please ensure you have an internet connection for the first download.")
        similarity_model = None

    # --- Mistral GGUF Model Initialization ---
    print(f"Initializing Mistral LLM from local GGUF file ({MISTRAL_MODEL_PATH})...")
    try:
        if not os.path.exists(MISTRAL_MODEL_PATH):
            raise FileNotFoundError(f"GGUF model file not found at {MISTRAL_MODEL_PATH}")

        # Load GGUF model using llama-cpp-python
        llm_model = Llama(
            model_path=MISTRAL_MODEL_PATH,
            n_ctx=2048,  # Context length
            n_threads=4,  # Number of CPU threads (adjust based on your system)
            n_gpu_layers=0,  # Set to >0 if you have a compatible GPU (e.g., NVIDIA with CUDA)
            verbose=False  # Reduce logging
        )
        print("Mistral GGUF LLM initialized.")
    except Exception as e:
        print(f"Error initializing Mistral GGUF LLM: {e}")
        print("Ensure the GGUF file exists and you have sufficient memory.")
        llm_model = None

    return similarity_model is not None and llm_model is not None

# --- Task 1: Chatbot Implementation ---
def generate_context_for_part(part_id):
    """Generates a textual context for a given part ID."""
    if parts_df is None:
        return "Data not loaded. Please check the data file."

    part_info = parts_df[parts_df['ID'].str.lower() == part_id.lower()]
    if part_info.empty:
        return None  # Indicate part not found

    part_data = part_info.iloc[0].dropna().to_dict()
    context = f"Information about Part ID {part_id}:\n"
    for key, value in part_data.items():
        if key not in ['ID', 'CLEANED_DESCRIPTION', 'EMBEDDING']:  # Exclude internal columns
            context += f"{key}: {value}. "
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
        if similarities[idx] < 0.7:  # Example threshold
            continue
        part_id = parts_df.iloc[idx]['ID']
        part_description = parts_df.iloc[idx]['DESCRIPTION']
        context += f"Part ID: {part_id}, Description: {part_description}, Similarity Score: {similarities[idx]:.4f}.\n"
        count += 1

    if count == 0:
        return "No highly similar parts found for your query in the database."
    return context

def generate_llm_response(user_query, context=""):
    """Generates a response using the local Mistral GGUF model."""
    global llm_model
    if llm_model is None:
        return "LLM model not loaded. Please check initialization."

    # Construct prompt for Mistral-7B-Instruct
    system_prompt = "You are a helpful assistant for a parts catalog. Answer questions accurately based *only* on the provided context. If the information is not in the context, state that you don't know or cannot find the information."
    if context:
        full_prompt = f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_query} [/INST]"
    else:
        full_prompt = f"<s>[INST] {system_prompt}\n\nUser Question: {user_query} [/INST]"

    try:
        # Generate response using llama-cpp-python
        response = llm_model(
            full_prompt,
            max_tokens=256,  # Max tokens for response
            temperature=0.7,  # Control randomness
            top_p=1.0,  # Nucleus sampling
            stop=["</s>"]  # Stop at end of instruction
        )
        # Extract the generated text
        return response['choices'][0]['text'].strip()
    except Exception as e:
        return f"An error occurred with LLM: {e}"

def chatbot_interface_llm():
    """Interactive chatbot leveraging LLM."""
    print("\n--- Task 1: Interface (with RAG capabilities) ---")
    print("Hello! I'm your Parts Information Chatbot. I can answer questions about parts and find similar ones.")
    print("Examples: 'What is the material of part A1?', 'Tell me about A9', 'Find parts similar to A10'.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        context = ""

        # 1. Try to find a specific part ID
        part_id_match = re.search(r'\b([aA]\d+)\b', user_input)
        if part_id_match:
            part_id = part_id_match.group(1).upper()
            part_context = generate_context_for_part(part_id)
            if part_context is None:
                context = f"Part with ID '{part_id}' not found in the database."
            else:
                context = part_context

        # 2. Check for "find similar" intent
        elif "similar" in user_input.lower() or "alternative" in user_input.lower():
            context = retrieve_similar_parts_context(user_input)
        else:
            context = ""

        # 3. Pass query and context to LLM
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
        'Joule-integral-Nom (J)']

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
        'Operating Temperature-Max (Cel)', 'Operating Temperature-Max (Cel)_Normalized']
    print(extracted_df[display_cols].head().to_string())
    print("\nSummary of Normalized Data (Non-null counts and Dtype):")
    print(extracted_df.info())
    return extracted_df

# --- Task 3: Similar Parts Finder ---
def preprocess_text_for_similarity(text):
    """Cleans description text for embedding."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s\-\.\/x]', '', text)  # Keep numbers, letters, spaces, -, ., /, x
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with single
    return text

def find_similar_parts(num_alternatives=5):
    print("\n--- Task 3: Similar Parts Finder ---")
    if parts_df is None or similarity_model is None or 'EMBEDDING' not in parts_df.columns:
        print("Data, similarity model, or pre-computed embeddings not loaded. Cannot find similar parts.")
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
        print("Exiting due to model initialization error. Please check dependencies and file paths.")
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