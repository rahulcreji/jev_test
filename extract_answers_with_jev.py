import os
import json
import time
from dotenv import load_dotenv
from snowflake.snowpark import Session
from typesafe_sdk import TypeSafeClient, Noul

os.environ["TYPESAFE_API_KEY"] = "apikey_21130796ec50d2d942adb9d027801ce72103_f2544a8e3067b354be5574371136df4afb3d491dbffff9971b35c55a8699310f"

# Database and Schema Setup
DATABASE = "CONFORMED"
SCHEMA = "HFS_FI_LEASE"
SCHEMA_PREFIX = f"{DATABASE}.{SCHEMA}"
TABLE_QUESTIONS_RULES = f"{SCHEMA_PREFIX}.LEASE_AI_QUESTIONS_RULES"
INT_STAGE_PATH = f"@{SCHEMA_PREFIX}.LEASE_AI_PDF_INT_STG"

def safe_val(val) -> str:
    """Return a SQL-safe string by escaping single quotes and backslashes or NULL.
    This prevents SQL injection and syntax termination errors, while preserving $$ strings.
    """
    if val is None:
        return "NULL"
    
    val_str = str(val)
    
    # Escape backslashes and single quotes for safe Snowflake single-quote literals
    val_str = val_str.replace("\\", "\\\\").replace("'", "''")
        
    return f"'{val_str}'"

def get_10_questions(session) -> dict:
    """Fetch exactly 10 questions from the questions table."""
    print(f"📥 Fetching 10 questions from {TABLE_QUESTIONS_RULES}...")
    query = f"SELECT ID, QUESTION FROM {TABLE_QUESTIONS_RULES} WHERE QUESTION IS NOT NULL ORDER BY ID LIMIT 10"
    rows = session.sql(query).collect()
    
    questions_dict = {}
    for row in rows:
        questions_dict[str(row["ID"])] = row["QUESTION"]
    return questions_dict

def load_ocr_and_get_paragraphs(session, filename: str) -> list:
    """Loads OCR JSON (using cache if available) and extracts small paragraphs."""
    safe_cache_name = "".join([c if c.isalnum() else "_" for c in filename]) + "_ocr_cache.json"
    cache_path = os.path.join(os.getcwd(), safe_cache_name)
    
    parsed_json_str = ""
    if os.path.exists(cache_path):
        print(f"📄 Found cached OCR locally! Loading from '{safe_cache_name}'...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            parsed_json_str = f.read()
    else:
        print(f"📄 Running OCR on real contract: '{filename}' from {INT_STAGE_PATH}...")
        safe_v = safe_val(filename)
        row = session.sql(f"""
            SELECT AI_PARSE_DOCUMENT(
                TO_FILE('{INT_STAGE_PATH}', {safe_v}),
                {{'mode': 'LAYOUT', 'page_split': TRUE}}
            ) AS PARSED_OUTPUT
        """).collect()[0]
        parsed_json_str = row["PARSED_OUTPUT"]
        
        print(f"💾 Saving OCR output to local cache '{safe_cache_name}'...")
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(parsed_json_str)
            
    print("✂️ Chunking the OCR output into paragraphs...")
    parsed_json = json.loads(parsed_json_str)
    
    # Extract text from pages
    if "pages" in parsed_json:
        full_text = "\n\n".join([p.get("content", "") for p in parsed_json["pages"]])
    elif "content" in parsed_json:
        full_text = parsed_json["content"]
    else:
        full_text = parsed_json_str
        
    # Split by double newline to get small paragraphs
    raw_paragraphs = full_text.split('\n\n')
    
    paragraphs = []
    for p in raw_paragraphs:
        clean_p = p.strip()
        # Only keep substantial paragraphs (e.g., > 50 characters) to avoid noise/headers/footers
        if len(clean_p) > 50:
            paragraphs.append(clean_p)
            
    print(f"✅ Extracted {len(paragraphs)} substantial paragraphs.")
    return paragraphs

def main():
    # 1. Connect to Snowflake Dev Environment
    load_dotenv()
    token = os.getenv("Dev_TOKEN")
    if not token:
        print("⚠️ Environment variable 'DEV_TOKEN' is not set.")
        return

    connection_parameters = {
        "user":      "v-rreji",  
        "password":  token,
        "account":   "HFSINCLAIR-DEV",
        "database":  "CONFORMED",
        "schema":    "HFS_FI_LEASE",
        "warehouse": "HFS_CUSTOM_WH",
        "role":      "DEVELOPER_FR",
    }

    print("🔌 Connecting to Snowflake Dev Environment...")
    session = Session.builder.configs(connection_parameters).create()
    
    # 2. Get 10 Questions
    questions_dict = get_10_questions(session)
    print(f"✅ Loaded {len(questions_dict)} questions.")
    
    # 3. Get Paragraphs from the PDF
    my_pdf_filename = "4 - Ek-Stream TCP 23 Dec 2025.pdf"
    paragraphs = load_ocr_and_get_paragraphs(session, my_pdf_filename)
    
    # 4. Setup Jev questions payload
    # Add industry-specific synonyms and context to help Jev catch implicit matches
    QUESTION_ENHANCEMENTS = {
        "1": "Look for a specific named vessel, serial number, or IMO number.",
        "2": "Look for phrases like 'exclusive use', 'whole reach', 'full capacity', or 'at Charterers disposal'.",
        "3": "Look for the right to issue voyage orders, dictate employment, trade limits, or cargo routing.",
        "4": "Look for operational control, master under charterer orders, or the right to operate the vessel.",
        "5": "Look for clauses stating the charterer designed or built the asset to their specifications.",
        "6": "Look for an 'option to purchase', 'option to buy', or 'right of first refusal'.",
        "7": "Look for 'transfer of title' or ownership passing to the charterer.",
        "8": "Look for references to the asset's 'useful life' or 'economic life'.",
        "9": "Look for 'present value', 'fair market value', or 'guaranteed residual value'.",
        "10": "Look for 'specialized nature', 'custom built', or 'no alternative use'."
    }
    
    jev_questions = {}
    for q_id, q_text in questions_dict.items():
        extra_context = QUESTION_ENHANCEMENTS.get(str(q_id), "")
        instructions = f"Does the following contract paragraph contain information that answers this question: '{q_text}'? {extra_context} Answer YES if it does."
        
        jev_questions[f"question_{q_id}"] = Noul(instructions=instructions)
        
    client = TypeSafeClient()
    
    # Result structure
    results = {
        q_id: {
            "question": q_text,
            "matching_paragraphs": []
        }
        for q_id, q_text in questions_dict.items()
    }
    
    print("\n🔍 Running Jev evaluation on all paragraphs (Threshold: > 80%)...")
    start_time = time.time()
    
    for i, para in enumerate(paragraphs):
        if i % 10 == 0:
            print(f"Processing paragraph {i}/{len(paragraphs)}...")
            
        pipeline_state = f"CONTRACT PARAGRAPH:\n{para}"
        
        try:
            response = client.system_one(state=pipeline_state, questions=jev_questions)
            
            # Check answers against the 80% threshold
            for key, answer in response.answers.items():
                if answer.noul >= 0.80:
                    q_id = key.replace("question_", "")
                    results[q_id]["matching_paragraphs"].append(para)
                    
        except Exception as e:
            print(f"Error processing paragraph {i}: {e}")
            
    exec_time = time.time() - start_time
    print(f"\n✅ Completed Jev evaluation in {exec_time:.2f} seconds.")
    
    # 5. Save final result to JSON file
    output_filename = "question_paragraph_matches.json"
    print(f"💾 Saving results to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print("🎉 Done! Check out question_paragraph_matches.json")

if __name__ == "__main__":
    main()
