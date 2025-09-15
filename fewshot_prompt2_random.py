import os
import time
import logging
import openai
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import random  # Import the random module for random selection

# --- CONFIGURATION ---
UNLABELED_DATASET_PATH = "tarannom-s/code_snippets" 
GOLDEN_DATASET_PATH = "tarannom-s/golden_dataset"
NUM_FEW_SHOTS = 4
OUTPUT_PATH = "./synthetic_rules/"
MODEL = "gpt-4o"

# --- SETUP ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
client = OpenAI(api_key=openai_api_key)

# --- LOADING THE GOLDEN DATASET ---
logging.info(f"Loading golden dataset from Hugging Face: {GOLDEN_DATASET_PATH}...")
golden_dataset = load_dataset(GOLDEN_DATASET_PATH, split="train")
golden_data = list(golden_dataset)
logging.info(f"Loaded {len(golden_data)} golden examples.")

# --- RANDOM RETRIEVAL FUNCTION ---
def retrieve_few_shots(k=NUM_FEW_SHOTS):
    """
    Randomly selects k examples from the golden dataset.
    
    Args:
        k (int): The number of few-shot examples to retrieve.
        
    Returns:
        list: A list of k randomly selected dictionaries, where each dictionary
              is an example from the golden dataset.
    """
    if len(golden_data) < k:
        logging.warning(f"Requested {k} few-shots, but only {len(golden_data)} available. Returning all.")
        return golden_data
    
    return random.sample(golden_data, k)

# --- PROMPT BUILDING FUNCTION ---
def build_prompt(few_shot_examples, current_code):
    """
    Constructs the prompt with a system message, few-shot examples,
    and the current code to analyze.
    """
    prompt_parts = [
        "You are a meticulous Code Analyzer and Documentation Expert. Your primary objective is to translate raw code logic into clear, human-readable business rules. The final output must be a well-structured markdown document."
    ]
    for i, ex in enumerate(few_shot_examples):
        prompt_parts.append(f"### Example {i+1}")
        prompt_parts.append(f"Code:\n{ex['DCx.txt']}\n")
        prompt_parts.append(f"Business Rule:\n{ex['BRx.md']}\n")
        prompt_parts.append("---")
    
    prompt_parts.append("### Current Code to Analyze")
    prompt_parts.append("The following code snippet contains specific business rules. Your task is to identify and describe these rules in the specified markdown format.")
    prompt_parts.append(f"Code:\n{current_code}\n")
    
    prompt_parts.append("---")
    prompt_parts.append("### Extracted Business Rule")
    prompt_parts.append("Provide the business rules here as a well-structured markdown document, following a format similar to the examples above.")
    
    return "\n".join(prompt_parts)

# --- MAIN EXECUTION ---
logging.info(f"Loading unlabeled code snippets from Hugging Face: {UNLABED_DATASET_PATH}...")
unlabeled_dataset = load_dataset(UNLABELED_DATASET_PATH, split="train")
unlabeled_data = list(unlabeled_dataset)
logging.info(f"Found {len(unlabeled_data)} code snippets to process.")

os.makedirs(OUTPUT_PATH, exist_ok=True)
failed_files = []

logging.info("Starting the synthetic rule generation process...")

# Randomly select few-shots once before the loop
few_shots = retrieve_few_shots(k=NUM_FEW_SHOTS)
logging.info(f"Selected {len(few_shots)} random examples for few-shot prompting.")

for item in tqdm(unlabeled_data, desc="Processing code snippets"):
    file_name = item['ID'] + '.md'
    current_code = item['text']
    
    prompt = build_prompt(few_shots, current_code)

    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts and summarizes business rules from code snippets. The output must be a markdown document."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            output = response.choices[0].message.content
            
            output_file_path = os.path.join(OUTPUT_PATH, file_name)
            with open(output_file_path, "w", encoding="utf-8") as out_f:
                out_f.write(output)
            
            break
        except openai.OpenAIError as e:
            if attempt < retries - 1:
                logging.warning(f"OpenAI API Error for {file_name}: {e}. Retrying in {2**(attempt+1)} seconds...")
                time.sleep(2**(attempt+1))
            else:
                logging.error(f"Failed to generate rule for {file_name} after {retries} attempts: {e}")
                failed_files.append(file_name)
        except Exception as e:
            logging.error(f"An unexpected error occurred with {file_name}: {e}")
            failed_files.append(file_name)
            break

logging.info("Process completed.")
if failed_files:
    logging.warning(f"The following files failed to process: {failed_files}")