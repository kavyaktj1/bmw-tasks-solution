'''
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install --upgrade transformers sentence-transformers accelerate
pip show transformers sentence-transformers accelerate
python main.py
'''

import pandas as pd
import numpy as np
import re
import os 
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#from huggingface_hub import InferenceClient

# Global Data and Model Initialization details
parts_df = None
similarity_model = None # For sentence embeddings, crucial for RAG
llm_tokenizer = None # For Mistral
llm_model = None     # For Mistral
llm_pipeline = None  # If using Hugging Face pipeline for Mistral

# Configuration
CSV_FILEPATH = "Parts.csv"
SIMILARITY_MODEL_NAME = 'all-MiniLM-L6-v2'
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # Or another Mistral variant

# preprocess the tex - Data Cleaning
def preprocess_text_for_similarity(text):
    """Cleans description text for embedding."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s\-\.\/x]', '', text) # Keep numbers, letters, spaces, -, ., /, x
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single
    return text

# Data Loading Function
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

# Model Initialization Function
def initialize_models():
    """Initializes the NLP models for similarity and the LLM."""
    global similarity_model,llm_tokenizer, llm_model, llm_pipeline
    print(f"Initializing Sentence Transformer Model ({SIMILARITY_MODEL_NAME})...")
    try:
        similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME)
        print("Sentence Transformer Model initialized.")
    except Exception as e:
        print(f"Error initializing Sentence Transformer model: {e}")
        print("Please ensure you have an internet connection for the first download.")
        similarity_model = None
    
    # --- Mistral Model Initialization (Choose one method) ---
    # Option A: Local loading via Hugging Face Transformers (requires powerful GPU and VRAM at least 16–24 GB for 7B models)
    # print(f"Initializing Mistral LLM ({MISTRAL_MODEL_NAME})... This requires significant resources.")
    # try:
    #     llm_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME)
    #     llm_model = AutoModelForCausalLM.from_pretrained(MISTRAL_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    #     llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=256, device_map="auto")
    #     print("Mistral LLM initialized.")
    # except Exception as e:
    #     print(f"Error initializing Mistral LLM: {e}")
    #     print("Consider using a smaller model, checking GPU availability, or using an API.")
    #     llm_tokenizer, llm_model, llm_pipeline = None, None, None

    # Option B: Placeholder for API integration (e.g., using 'requests' library)
    # For this example, we'll assume a local Mistral setup or a very simple mock for demonstration.
    # A full API integration would involve API keys, endpoint URLs, and error handling for network requests.
    # For now, we'll keep the RAG logic, and you'd replace the `generate_llm_response` with actual API calls.

    return similarity_model is not None # and llm_pipeline is not None # If LLM is mandatory

# Task 1.1: Function to find a specific part ID
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

# Task 1.2 - Function to check for "find similar" intent (simple keyword check for demonstration)
def retrieve_similar_parts_context(query_text, num_results=3):
    """Retrieves context for similar parts based on a natural language query.
    This will use the pre-computed embeddings."""

    if similarity_model is None or 'EMBEDDING' not in parts_df.columns:
        return "Similarity search not available: model or embeddings missing."

    query_embedding = similarity_model.encode([preprocess_text_for_similarity(query_text)])
    similarities = cosine_similarity(query_embedding, np.array(parts_df['EMBEDDING'].tolist()))[0]
    #calculates cosine similarity between a query embedding and a list of stored embeddings in a DataFrame.
    #It tells you how similar the query is to each row in the dataset 

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

# Task 1.3 - Function to pas query and context to LLM for final answer
def generate_llm_response(user_query, context=""):
    """
    Generates a response using Mistral, leveraging provided context.
    This is a placeholder for actual LLM API/local inference.
    """

    # if llm_pipeline is None: # For local transformers pipeline
    #     return "LLM model not loaded. Please check initialization."

    # Construct the prompt for Mistral
    # Mistral-Instruct models typically use specific chat formats.
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
    # try:
    #     response = llm_pipeline(full_prompt)[0]['generated_text']
    #     # Extract only the model's answer part if the pipeline returns the full prompt + answer
    #     if "[/INST]" in response:
    #         response = response.split("[/INST]")[-1].strip()
    #     return response
    # except Exception as e:
    #     return f"An error occurred with the LLM: {e}"

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

# Task 1 Function
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

        # 1.1 Try to find a specific part ID
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

        # 1.2 Check for "find similar" intent (simple keyword check for demonstration)
        elif "similar" in user_input.lower() or "alternative" in user_input.lower():
            context = retrieve_similar_parts_context(user_input)
        else:
            # If no specific part ID or "similar" intent, still try to answer generally
            context = "" # No specific context from data retrieval initially

        # 1.3. Pass query and context to LLM for final answer
        response = generate_llm_response(user_input, context)
        print(f"Chatbot: {response}")

# Task 2 Function to find alternate parts
def find_similar_parts(num_alternatives=5):
    #Finds the top N most similar parts based on their description using pre-computed embeddings.
    print("\n--- Task 2: Similar Parts Finder ---")

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

    for part_data in results[:5]: # Print only first 5 for demonstration
        print(f"\nOriginal Part ID: {part_data['Original_ID']}")
        print(f"Description: \"{part_data['Original_Description']}\"")
        print("Found Alternatives:")
        if part_data['Alternatives']:
            for alt in part_data['Alternatives']:
                print(f"  - ID: {alt['Alternative_ID']}, Similarity: {alt['Similarity_Score']:.4f}, Description: \"{alt['Alternative_Description']}\"")
        else:
            print("  No alternatives found (or not enough data).")
    if len(results) > 5:
        print(f"\n... (Showing only first 5 results out of {len(results)}. Run this task again to see more.)")

    return results

# Main Execution Flow
def main():
    """Main function to orchestrate the execution of all tasks."""
    print("Starting Parts Analysis Application...")

    if not os.path.exists(CSV_FILEPATH):
        print(f"Error: {CSV_FILEPATH} not found. Please create it with 1000 records.")
        return

    # Initialize models first, then load data (which also uses models for embeddings)
    if not initialize_models():
        print("Exiting due to model initialization error. Please check dependencies and internet connection.")
        return
    
    # Load data and precompute embeddings
    if not load_data(): 
        print("Exiting due to data loading error.")
        return

    while True:
        print("\nHello, Hope you are doing good. Select a task to run:")
        print("1. LLM Chatbot Interface (Task 1)")
        print("2. Similar Parts Finder (Task 3)")
        print("0. Exit Application")

        choice = input("Enter your choice (0-2): ").strip()
        if choice == '1':
            chatbot_interface_llm()

        elif choice == '2':
            find_similar_parts()

        elif choice == '0':
            print("Exiting application. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 0 and 3.")

if __name__ == "__main__":
    main()