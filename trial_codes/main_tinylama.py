import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os # For checking file existence

# --- LLM Integration specific imports ---
# Local via Hugging Face Transformers (requires significant VRAM for larger models)
# Use transformers with Mistral-7B-Instruct from Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# --- Global Data and Model Initialization ---
parts_df = None
similarity_model = None # For sentence embeddings, crucial for RAG
llm_tokenizer = None # For Mistral
llm_model = None     # For Mistral
llm_pipeline = None  # If using Hugging Face pipeline for Mistral

# --- Configuration ---
CSV_FILEPATH = "Parts.csv"
SIMILARITY_MODEL_NAME = 'all-MiniLM-L6-v2'
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Or another Mistral variant

# --- 1. Data Loading ---
def load_data(filepath=CSV_FILEPATH):
    """Loads the parts data from a CSV file."""
    global parts_df
    try:
        parts_df = pd.read_csv(filepath, sep=';')
        parts_df.replace('', np.nan, inplace=True) # Replace empty strings with NaN
        print(f"Data loaded successfully from {filepath}. Shape: {parts_df.shape}")

        # Precompute cleaned descriptions and embeddings for faster similarity search
        parts_df.dropna(subset=['DESCRIPTION'], inplace=True)
        parts_df['CLEANED_DESCRIPTION'] = parts_df['DESCRIPTION'].apply(preprocess_text_for_similarity)
        print("Preprocessing descriptions for similarity...")

        global similarity_model # Ensure model is loaded before embedding

        if similarity_model is None:
            initialize_models() # Initialize if not already

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
'''
def initialize_models():
    """Initializes the NLP models for similarity and the LLM."""
    global similarity_model , llm_tokenizer, llm_model, llm_pipeline
    print(f"Initializing Sentence Transformer Model ({SIMILARITY_MODEL_NAME})...")
    try:
        similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME)
        print("Sentence Transformer Model initialized.")
    except Exception as e:
        print(f"Error initializing Sentence Transformer model: {e}")
        print("Please ensure you have an internet connection for the first download.")
        similarity_model = None

    # --- Mistral Model Initialization (Choose one method) ---
    # Option A: Local loading via Hugging Face Transformers (requires powerful GPU and VRAM)

    print(f"Initializing Mistral LLM ({MISTRAL_MODEL_NAME})... This requires significant resources.")
    try:
        from huggingface_hub import login
        login(token="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")  # Replace with your token
        # Load tokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)

        # Load model with 4-bit quantization for efficiency
        # load_in_4bit=True with bitsandbytes reduces VRAM usage to ~7-8GB
        llm_model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL_NAME,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for lower memory usage
            device_map="auto",  # Automatically distribute across GPU/CPU, uses accelerate
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computations
            bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
            bnb_4bit_use_double_quant=True  # Double quantization for better efficiency
        )

        # Create pipeline for text generation
        llm_pipeline = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=256,  # Max tokens for response
            device_map="auto"
        )

        print("Mistral LLM initialized.")
    except Exception as e:
        print(f"Error initializing Mistral LLM: {e}")
        print("Consider using a smaller model, checking GPU availability, or using an API.")
        llm_tokenizer, llm_model, llm_pipeline = None, None, None

    return similarity_model is not None and llm_pipeline is not None
'''

def initialize_models():
    """Initializes the NLP models for similarity and the LLM."""
    global similarity_model, llm_tokenizer, llm_model, llm_pipeline
    print(f"Initializing Sentence Transformer Model ({SIMILARITY_MODEL_NAME})...")
    try:
        similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME, device='mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Sentence Transformer Model initialized on {'MPS' if torch.backends.mps.is_available() else 'CPU'}.")
    except Exception as e:
        print(f"Error initializing Sentence Transformer model: {e}")
        similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME, device='cpu')
        print("Sentence Transformer Model initialized on CPU (fallback).")

    # --- LLM Initialization (Optimized for Mac M1) ---
    print("Initializing LLM (TinyLlama-1.1B for Mac M1)...")
    try:
        from huggingface_hub import login
        login(token="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")  # Your token

        # Use TinyLlama-1.1B (1.1B parameters, ~2-3 GB RAM)
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # FP16 for M1
            device_map="auto",  # Use CPU
            low_cpu_mem_usage=True  # Minimize memory usage
        )
        llm_pipeline = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_tokenizer,
            max_new_tokens=128,  # Low for memory efficiency
            #device="mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"TinyLlama-1.1B initialized on {'MPS' if torch.backends.mps.is_available() else 'CPU'}.")

    except Exception as e:
        print(f"Error initializing TinyLlama LLM: {e}")
        print("Falling back to Hugging Face Inference API for Mistral-7B...")
        #We’ll also keep the Hugging Face Inference API as a fallback if local inference fails.
        try:
            from huggingface_hub import InferenceClient
            llm_pipeline = InferenceClient(
                model=MISTRAL_MODEL_NAME,  # mistralai/Mistral-7B-Instruct-v0.2
                token="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"
            )
            print("Hugging Face Inference API initialized for Mistral-7B.")
        except Exception as e:
            print(f"Error initializing Inference API: {e}")
            llm_tokenizer, llm_model, llm_pipeline = None, None, None

    return similarity_model is not None and llm_pipeline is not None

# --- Task 1: Chatbot Implementation (Revised for LLM) ---
def generate_context_for_part(part_id):
    """Generates a textual context for a given part ID."""
    if parts_df is None:
        return "Data not loaded. Please check the data file."

    # Use .loc for direct ID lookup, it's efficient even with 1000 IDs
    part_info = parts_df[parts_df['ID'].str.lower() == part_id.lower()]

    if part_info.empty:
        return None # Indicate part not found

    part_data = part_info.iloc[0].dropna().to_dict()
    #part_data : {'ID': 'A10', 'DESCRIPTION': 'Non Resettable Indicators Electric Indicator, Time Lag Blow, 2A, 250VAC, 300VDC, 1500A (IR), Inline/holder, 5x20mm', 'Additional Feature': 'RATED BREAKING CAPACITY AT 300 VDC: 1500 A'}
    context = f"Information about Part ID {part_id}:\n"
    for key, value in part_data.items():
        if key not in ['ID', 'CLEANED_DESCRIPTION', 'EMBEDDING']: # Exclude internal columns
            context += f"{key}: {value}. "

    return context.strip()

def retrieve_similar_parts_context(query_text, num_results=3):
    """Retrieves context for similar parts based on a natural language query.
    This will use the pre-computed embeddings."""

    if similarity_model is None or 'EMBEDDING' not in parts_df.columns:
        return "Similarity search not available: model or embeddings missing."

    query_embedding = similarity_model.encode([preprocess_text_for_similarity(query_text)])
    similarities = cosine_similarity(query_embedding, np.array(parts_df['EMBEDDING'].tolist()))[0]

    # Get indices of top N similar parts (excluding self-similarity if query is an existing part description)
    # Sort in descending order and take top N, make sure not to include the query itself if it's in the dataset
    top_indices = similarities.argsort()[::-1]

    # Build context string
    context = "Here is some information about potentially similar parts:\n"
    count = 0
    for idx in top_indices:
        if count >= num_results:
            break

        # Optional: Add a threshold to only include highly similar items
        if similarities[idx] < 0.7: # Example threshold
            continue

        part_id = parts_df.iloc[idx]['ID']
        part_description = parts_df.iloc[idx]['DESCRIPTION']
        context += f"Part ID: {part_id}, Description: {part_description}, Similarity Score: {similarities[idx]:.4f}.\n"
        count += 1

    if count == 0:
        return "No highly similar parts found for your query in the database."

    return context
'''
def generate_llm_response(user_query, context=""):
    """
    Generates a response using Mistral, leveraging provided context.
    This is a placeholder for actual LLM API/local inference.
    """
    global llm_pipeline
    
    # For local transformers pipeline
    if llm_pipeline is None: 
        return "LLM model not loaded. Please check initialization."

    # Construct the prompt for Mistral - Mistral-Instruct models typically use specific chat formats.
    # Example for Mistral-7B-Instruct-v0.2:
    # <s>[INST] Instruction [/INST] Model answer</s>
    # <s>[INST] Follow-up instruction [/INST] Model follow-up answer</s>

    system_prompt = "You are a helpful assistant for a parts catalog. Answer questions accurately based *only* on the provided context. If the information is not in the context, state that you don't know or cannot find the information."

    if context:
        full_prompt = f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_query} [/INST]"
    else:
        full_prompt = f"<s>[INST] {system_prompt}\n\nUser Question: {user_query} [/INST]"

    # --- This is where you'd call your Mistral model ---
    # Option A: Local Hugging Face pipeline
    try:
        # Generate response using pipeline
        response = llm_pipeline(full_prompt)[0]['generated_text']
        # Extract only the model's answer part if the pipeline returns the full prompt + answer
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        return response
    except Exception as e:
        return f"An error occurred with the LLM: {e}"

    # Option B: Mock LLM response for demonstration without actual Mistral setup

    if "material" in user_query.lower() and "ceramic" in context.lower():
        return "Based on the provided information, the material appears to be Ceramic."

    elif "current" in user_query.lower() and "1.6A" in context.lower():
         return "The rated current for that part is 1.6A according to the data."

    elif "no highly similar parts" in context.lower():
        return "I couldn't find very similar parts based on your query in the database."

    elif "not found" in context:
        return context # Pass through the "part not found" message

    elif context and "Information about Part ID" in context:
        return f"Here is what I found:\n{context}\nIf you have a specific question, please ask."

    elif "temperature" in user_query.lower() and "85Cel" in context.lower():
        return "The maximum operating temperature for that part is 85 Celsius, and the minimum is -40 Celsius."

    else:
        return "I'm not sure how to answer that based on the provided data. Could you rephrase or ask about a specific part ID or characteristic?"
'''

def generate_llm_response(user_query, context=""):
    """
    Generates a response using TinyLlama or Hugging Face Inference API, leveraging provided context.
    """
    global llm_pipeline
    
    if llm_pipeline is None: 
        return "LLM model not loaded. Please check initialization."

    # Construct the prompt for TinyLlama or Mistral
    system_prompt = "You are a helpful assistant for a parts catalog. Answer questions accurately based *only* on the provided context. If the information is not in the context, state that you don't know or cannot find the information."
    #full_prompt = f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_query} [/INST]"
    full_prompt = f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nUser: {user_query} [/INST]"

    try:
        # Check input token length
        inputs = llm_tokenizer(full_prompt, return_tensors="pt")
        token_length = inputs.input_ids.shape[-1]
        print(f"Input token length: {token_length}")
        if token_length > 2048:
            return "Input too long for the model. Please shorten the query or context."

        # Check if using InferenceClient (API) or pipeline (local)
        if isinstance(llm_pipeline, InferenceClient):
            response = llm_pipeline.text_generation(full_prompt, max_new_tokens=128)
        else:
            print("----------generating response--------")
            raw_response = llm_pipeline(full_prompt, max_new_tokens=128, do_sample=True, top_p=0.9)[0]['generated_text']
            print(f"Raw response: {raw_response}")
            print("-------------------------------")

            # Remove prompt from response if present
            response = raw_response[len(full_prompt):].strip() if raw_response.startswith(full_prompt) else raw_response
        print("*********************************************")
        return response if response else "LLM returned an empty response."
    except Exception as e:
        return f"An error occurred with the LLM: {e}"

def chatbot_interface_llm():
    """Interactive chatbot leveraging an LLM for answering questions."""

    print("\n--- Task 1: LLM Chatbot Interface (with RAG capabilities) ---")
    print("Hello! I'm your Parts Information Chatbot. I can answer questions about parts and find similar ones.")
    print("Examples: 'What is the material of part A1?', 'Tell me about A9', 'Find parts similar to A10'.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        #user_input : tell me about A10
        context = ""

        # 1. Try to find a specific part ID
        part_id_match = re.search(r'\b([aA]\d+)\b', user_input)
        #part_id_match : <re.Match object; span=(14, 17), match='A10'>
        if part_id_match:
            part_id = part_id_match.group(1).upper()
            #part_id : A10
            part_context = generate_context_for_part(part_id)
            if part_context is None: # Part not found
                context = f"Part with ID '{part_id}' not found in the database."
            else:
                context = part_context

        # 2. Check for "find similar" intent (simple keyword check for demonstration)
        elif "similar" in user_input.lower() or "alternative" in user_input.lower():
            context = retrieve_similar_parts_context(user_input)
        else:
            # If no specific part ID or "similar" intent, still try to answer generally
            context = "" # No specific context from data retrieval initially

        # 3. Pass query and context to LLM for final answer
        response = generate_llm_response(user_input, context)
        print(f"Chatbot: {response}")

# --- Rest of the code (Task 2 & 3 functions) remain largely the same or are refined ---
# --- Task 2: Data Extraction and Normalization (Unchanged in logic, just callable) ---

def extract_and_normalize_data():
    # ... (Keep the existing Task 2 code as is) ...
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

# --- Task 3: Similar Parts Finder (Modified to use pre-computed embeddings) ---

def preprocess_text_for_similarity(text):
    """Cleans description text for embedding."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s\-\.\/x]', '', text) # Keep numbers, letters, spaces, -, ., /, x
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single
    return text


def find_similar_parts(num_alternatives=5):
    #Finds the top N most similar parts based on their description using pre-computed embeddings.
    print("\n--- Task 3: Similar Parts Finder ---")

    if parts_df is None or similarity_model is None or 'EMBEDDING' not in parts_df.columns:
        print("Data, similarity model, or pre-computed embeddings not loaded. Cannot find similar parts.")
        print("Please ensure `load_data()` ran successfully and initialized embeddings.")
        return

    results = []

    # Ensure we use the index that corresponds to the 'EMBEDDING' series
    # Using iterrows is generally slower for large DFs but fine for iterating for display.
    # For computation, we use the pre-computed 'EMBEDDING' column.

    description_embeddings = np.array(parts_df['EMBEDDING'].tolist()) # Convert list of arrays to 2D numpy array
    print("Calculating cosine similarity matrix...")

    similarity_matrix = cosine_similarity(description_embeddings)
    print("Similarity matrix calculated.")

    for i, (original_idx, row) in enumerate(parts_df.iterrows()): # Use original_idx to get the original row
        original_id = row['ID']
        original_description = row['DESCRIPTION']
        # Get similarity scores for the current part, excluding itself

        similar_indices_in_matrix = similarity_matrix[i].argsort()[::-1][1:num_alternatives+1]
        alternative_parts = []

        for sim_idx_in_matrix_array in similar_indices_in_matrix:
            # Map back to the DataFrame's internal sequential index if needed, or use iloc
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
    # For large datasets, printing all results might be overwhelming.
    # You might want to print only the first few or allow user to search for specific parts.

    for part_data in results[:10]: # Print only first 10 for demonstration
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

        # Optionally, run the dummy data generator here if it's for testing
        # from your_generator_script import generate_dummy_parts_csv
        # generate_dummy_parts_csv(num_rows=1000, filename=CSV_FILEPATH)
        # if not load_data(): return # Try loading again after generating
        return

    # Initialize models first, then load data (which also uses models for embeddings)
    if not initialize_models():
        print("Exiting due to model initialization error. Please check dependencies and internet connection.")
        return

    if not load_data(): # Load data and precompute embeddings
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


'''
conda deactivate
conda env remove -n bmw_chatbot
conda create -n bmw_chatbot python=3.11
conda activate bmw_chatbot
conda install pandas=2.3.0 numpy=1.26.4 scikit-learn=1.5.1 sentence-transformers transformers pytorch accelerate
conda config --add channels conda-forge
pip install pandas==2.3.0 numpy==1.26.4 scikit-learn==1.5.1 sentence-transformers==3.0.1 transformers==4.44.2 torch==2.4.1 accelerate==0.33.0
python -c "import numpy; import pandas; import sklearn; import sentence_transformers; import transformers; import accelerate; import torch; print('No errors!')"
cd /Users/kavyakt/Downloads/BMW_GenAI_Chatbot
pip install bitsandbytes==0.43.0

conda activate bmw_chatbot
cd /Users/kavyakt/Downloads/BMW_GenAI_Chatbot
python main.py

'''
