import os
import time
import logging
import openai
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ast, astor

# --- CONFIGURATION ---
# The dataset containing the unlabeled code snippets to analyze
UNLABELED_DATASET_PATH = "tarannom-s/code_snippets" 

# The dataset containing the golden examples for few-shot prompting
GOLDEN_DATASET_PATH = "tarannom-s/golden_dataset"

# The number of most relevant examples to use for few-shot prompting
NUM_FEW_SHOTS = 4

# The output path for the generated business rules
OUTPUT_PATH = "./synthetic_rules/"

# The OpenAI model to use
MODEL = "gpt-4o"

# --- SETUP ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
client = OpenAI(api_key=openai_api_key)

# --- AST NORMALIZER ---
class NormalizeNames(ast.NodeTransformer):
    """
    Normalizes variable, function, and class names in a code's Abstract Syntax Tree (AST)
    to help the embedding model focus on the code's structure and logic,
    rather than specific naming conventions.
    """
    def __init__(self):
        super().__init__()
        self.var_map = {}
        self.counter = 0

    def _new_name(self, base="var"):
        self.counter += 1
        return f"{base}{self.counter}"

    def visit_Name(self, node):
        if node.id not in self.var_map:
            self.var_map[node.id] = self._new_name()
        node.id = self.var_map[node.id]
        return node

    def visit_arg(self, node):
        if node.arg not in self.var_map:
            self.var_map[node.arg] = self._new_name("arg")
        node.arg = self.var_map[node.arg]
        return node

    def visit_FunctionDef(self, node):
        node.name = "func"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        node.name = "Class"
        self.generic_visit(node)
        return node

def normalize_code(code_str: str) -> str:
    try:
        tree = ast.parse(code_str)
        normalizer = NormalizeNames()
        tree = normalizer.visit(tree)
        return astor.to_source(tree)
    except Exception:
        return code_str 

# --- EMBEDDING AND FAISS SETUP ---
logging.info(f"Loading golden dataset from Hugging Face: {GOLDEN_DATASET_PATH}...")
golden_dataset = load_dataset(GOLDEN_DATASET_PATH, split="train")
golden_data = list(golden_dataset)
logging.info(f"Loaded {len(golden_data)} golden examples.")

logging.info("Building embeddings for golden dataset...")
embed_model = SentenceTransformer("microsoft/graphcodebert-base")

# Use the 'DCx.txt' column for code normalization and embedding
normalized_codes = [normalize_code(item["DCx.txt"]) for item in golden_data]
embeddings = embed_model.encode(normalized_codes, convert_to_numpy=True, normalize_embeddings=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
logging.info("FAISS index built successfully.")

# --- RETRIEVAL FUNCTION ---
def retrieve_few_shots(query_code, k=NUM_FEW_SHOTS):
    """
    Retrieves the k most relevant few-shot examples from the golden dataset.
    """
    query_norm = normalize_code(query_code)
    query_emb = embed_model.encode([query_norm], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_emb, k)
    
    # Return both the code and the business rule for each relevant example
    return [golden_data[i] for i in idxs[0]]

# --- PROMPT BUILDING FUNCTION ---
def build_prompt(few_shot_examples, current_code):
    """
    Constructs the prompt with a system message, few-shot examples,
    and the current code to analyze, using the enhanced prompt.
    """
    system_prompt = (
        "You are a meticulous Code Analyzer and Documentation Expert. "
        "Your primary objective is to translate raw code logic into clear, human-readable business rules. "
        "The final output must be a well-structured markdown document."
    )

    # Building the user message with the full, detailed instructions
    user_prompt_parts = [
    """You are a meticulous **Code Analyzer** and **Documentation Expert**. Your primary objective is to translate raw code logic into clear, human-readable **business rules**. You will be provided with 3-5 few-shot examples, each containing a code snippet and its corresponding business rules.

Based on these examples, you must perform the following tasks:

1.  **Analyze and Learn:**
    * Thoroughly examine the provided few-shot examples to understand the distinction between a business rule and a technical implementation detail. Notice what elements from the code are excluded from the business rules (e.g., variable declarations, function names, database queries, API endpoints).
    * Absorb the specific writing style, tone, and formatting used in the example business rules. Your final output must replicate this learned format precisely.

2.  **Process New Code:**
    * You will be given a new piece of code (Code A).
    * Apply the knowledge gained from the few-shot examples to identify all a priori business rules embedded within Code A.
    * **Crucially, filter out all technical details.** A business rule must describe *why* a function or piece of logic exists, not *how* it is technically implemented. For example, "A user's account must be active to log in" is a business rule, while "Check for `isActive` boolean in the `users` table" is a technical detail.

3.  **Adhere to Business Rule Characteristics and Structure:**
    * **Atomic:** Each rule must represent a single, indivisible concept.
    * **Testable:** The rule must be verifiable. It should be possible to create a test case that proves or disproves the rule's validity.
    * **Clear and Unambiguous:** The language used must be easily understood by non-technical stakeholders. Avoid any ambiguity or technical jargon.
    * **Concise and Non-Repetitive:** While striving for atomicity, **avoid writing repetitive sentences** where only a single word or value changes. Instead, group related, atomic rules under a single, consolidated heading or use a list format to highlight variations. For example, instead of "A product must have a price," "A product must have a color," write "A product must have the following properties: Price, Color, Size."

4.  **Self-Review and Finalization:**
    * Before generating the final output, review your drafted business rules.
    * Ask yourself the following questions:
        * "Does this rule describe a business outcome or a technical action?"
        * "Can a non-technical person understand this without a code background?"
        * "Is this rule concise and non-repetitive?"
    * Make any necessary edits to ensure the output is of the highest quality and adheres strictly to the persona of a documentation expert.

5.  **Generate Output:**
    * Create a professional markdown document.
    * The document's title should be "**Business Rules for Code A**."
    * Each rule should be presented clearly and concisely, following the learned format.
    * Do not include any technical jargon, code snippets, or implementation-specific information in the final document.

Your goal is to produce a document that is easily understood by non-technical stakeholders, effectively serving as a guide to the system's core business logic while adhering to the principles of a well-formed business rule.
"""
]

    for i, ex in enumerate(few_shot_examples):
        prompt_parts.append(f"### Example {i+1}")
        # Use the 'DCx.txt' and 'BRx.md' columns from the golden dataset
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
logging.info(f"Loading unlabeled code snippets from Hugging Face: {UNLABELED_DATASET_PATH}...")
unlabeled_dataset = load_dataset(UNLABELED_DATASET_PATH, split="train")
unlabeled_data = list(unlabeled_dataset)
logging.info(f"Found {len(unlabeled_data)} code snippets to process.")

os.makedirs(OUTPUT_PATH, exist_ok=True)
failed_files = []

logging.info("Starting the synthetic rule generation process...")

for item in tqdm(unlabeled_data, desc="Processing code snippets"):
    file_name = item['ID'] + '.md' # Assuming a logical way to name output files
    current_code = item['text']

    few_shots = retrieve_few_shots(current_code, k=NUM_FEW_SHOTS)
    prompt = build_prompt(few_shots, current_code)

    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
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