import os
import json
import time
import csv

# 1. TypeSafe SDK for Jev
from typesafe_sdk import TypeSafeClient, Noul
from snowflake.snowpark import Session
from dotenv import load_dotenv

# ==========================================
# CONSTANTS & UTILITIES (No external imports)
# ==========================================

INT_STAGE_PATH = "@CONFORMED.HFS_FI_LEASE.LEASE_AI_PDF_INT_STG"

# Ensure Jev API key is set
os.environ["TYPESAFE_API_KEY"] = "apikey_21130796ec50d2d942adb9d027801ce72103_f2544a8e3067b354be5574371136df4afb3d491dbffff9971b35c55a8699310f"

def safe_val(val) -> str:
    """Return a SQL-safe string by escaping single quotes and backslashes."""
    if val is None:
        return "NULL"
    val_str = str(val)
    val_str = val_str.replace("\\", "\\\\").replace("'", "''")
    return f"'{val_str}'"

def process_and_accumulate_pages(session, parsed_json_str: str, chunk_size: int = 150000) -> list:
    """
    Parse the JSON output from AI_PARSE_DOCUMENT, add page numbers, 
    and accumulate into chunks.
    """
    layout_json = json.loads(parsed_json_str)
    pages = layout_json.get("pages", [])
    
    final_results = []
    current_chunk_text = ""
    
    for page_idx, page in enumerate(pages):
        page_num = page_idx + 1
        page_text = page.get("content", "")
        formatted_page = f"<page number--{page_num}>\n{page_text}"
        
        if len(current_chunk_text) + len(formatted_page) <= chunk_size:
            if current_chunk_text:
                current_chunk_text += "\n\n" + formatted_page
            else:
                current_chunk_text = formatted_page
        else:
            if current_chunk_text:
                final_results.append(current_chunk_text)
            current_chunk_text = formatted_page
            
    if current_chunk_text:
        final_results.append(current_chunk_text)
        
    return final_results


# ==========================================
# OCR & AI LOGIC
# ==========================================

def get_real_ocr_chunks(session, filename: str) -> list:
    """Runs the REAL OCR process on a PDF from your internal stage (with local caching)."""
    
    # Create a safe filename for the local cache
    safe_cache_name = "".join([c if c.isalnum() else "_" for c in filename]) + "_ocr_cache.json"
    cache_path = os.path.join(os.getcwd(), safe_cache_name)
    
    if os.path.exists(cache_path):
        print(f"📄 Found cached OCR locally! Loading from '{safe_cache_name}' to save time...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            parsed_json_str = f.read()
    else:
        print(f"📄 Running OCR on real contract: '{filename}' from {INT_STAGE_PATH}...")
        row = session.sql(f"""
            SELECT AI_PARSE_DOCUMENT(
                TO_FILE('{INT_STAGE_PATH}', {safe_val(filename)}),
                {{'mode': 'LAYOUT', 'page_split': TRUE}}
            ) AS PARSED_OUTPUT
        """).collect()[0]
        
        parsed_json_str = row["PARSED_OUTPUT"]
        
        # Save to local cache for future runs
        print(f"💾 Saving OCR output to local cache '{safe_cache_name}'...")
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(parsed_json_str)
    
    print("✂️ Chunking the OCR output...")
    chunk_size_chars = 150000
    chunks = process_and_accumulate_pages(session, parsed_json_str, chunk_size_chars)
    
    return chunks

def flatten_registry_to_jev_questions(registry_data: dict) -> dict:
    questions_payload = {}
    def process_node(node_id, node_data):
        title = node_data.get("title", "")
        highlights = " ".join(node_data.get("highlights", []))
        rule_desc = f"{title}. {highlights}"
        questions_payload[f"section_{node_id}"] = Noul(
            instructions=f"Does the rule or illustrative example defined as '{rule_desc}' apply to the provided CONTRACT TEXT? If it is an example (e.g. C1, C2), answer YES if the contract's scenario relates to it."
        )
        if "sub_sections" in node_data:
            for sub_id, sub_data in node_data["sub_sections"].items():
                process_node(sub_id, sub_data)

    for top_id, top_data in registry_data.items():
        process_node(top_id, top_data)
        
    return questions_payload


def evaluate_with_jev(contract_chunk: str, registry_data: dict, questions_dict: dict) -> dict:
    """Evaluates using Jev and returns metrics dictionary."""
    print("\n" + "="*50)
    print("🚀 RUNNING JEV MODEL EVALUATION (NOUL)")
    print("="*50)
    
    start_time = time.time()
    client = TypeSafeClient()
    pipeline_state = f"""You are an expert lease accounting AI evaluating a contract against ASC 842 rules.
TARGET QUESTIONS:
{json.dumps(questions_dict)}

CONTRACT TEXT:
{contract_chunk}
"""
    questions_payload = flatten_registry_to_jev_questions(registry_data)
        
    try:
        response = client.system_one(state=pipeline_state, questions=questions_payload)
        end_time = time.time()
        
        applicable = [key.replace("section_", "") for key, answer in response.answers.items() if answer.noul >= 0.80]
                
        input_tokens = response.usage.input_tokens
        output_tokens = 0 # Jev does not output generative tokens
        
        # Jev Pricing: $0.042 per 1M input tokens, Output is $0
        input_cost = (input_tokens / 1_000_000) * 0.042
        output_cost = 0.0
        total_cost = input_cost + output_cost
        
        exec_time = end_time - start_time
        
        print(f"✅ Found Applicable Sections: {applicable}")
        print(f"⏱️ Time Taken: {exec_time:.2f} seconds")
        print(f"💰 Jev Input Tokens: {input_tokens:,} | Output Tokens: {output_tokens}")
        print(f"💵 Jev Total Cost: ${total_cost:.6f}")
        
        return {
            "Model": "TypeSafe Jev (System One)",
            "Execution Time (s)": round(exec_time, 2),
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens,
            "Input Cost (USD)": input_cost,
            "Output Cost (USD)": output_cost,
            "Total Cost (USD)": total_cost,
            "Sections": applicable
        }
        
    except Exception as e:
        print(f"Jev API Error: {e}")
        return None

def evaluate_with_gpt_cortex(session, contract_chunk: str, registry_data: dict, questions_dict: dict) -> dict:
    """Evaluates using GPT-5 Cortex and returns metrics dictionary."""
    print("\n" + "="*50)
    print("🧠 RUNNING GPT-5 (CORTEX) EVALUATION")
    print("="*50)
    
    start_time = time.time()
    
    schema_routing = {
        "type": "object",
        "properties": {
            "applicable_sections": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["applicable_sections"],
        "additionalProperties": False
    }
    
    prompt = f"""You are an expert lease accounting AI evaluating a contract against ASC 842 rules.
TARGET QUESTIONS:
{json.dumps(questions_dict)}

ACCOUNTING RULES REGISTRY:
{json.dumps(registry_data)}

CONTRACT TEXT:
{contract_chunk}

INSTRUCTIONS:
Identify which sections from the REGISTRY apply to the provided CONTRACT TEXT based on the TARGET QUESTIONS.
Output ONLY the list of applicable section IDs.
Some sections (e.g., C1, C2) contain illustrative examples; you should also map these example sections if the contract's scenario relates to the example.
"""

    safe_prompt = prompt.replace("'", "''")
    safe_schema = json.dumps(schema_routing).replace("'", "''")
    
    ai_sql = f"""
    WITH model_inference AS (
        SELECT 
            '{safe_prompt}' AS user_prompt,
            SNOWFLAKE.CORTEX.AI_COMPLETE(
                model => 'openai-gpt-5',
                prompt => '{safe_prompt}',
                model_parameters => PARSE_JSON('{{"temperature": 0}}'),
                response_format => PARSE_JSON('{{"type": "json", "schema": {safe_schema}}}')
            ) AS ai_response_raw
    ),
    token_counting AS (
        SELECT 
            user_prompt,
            ai_response_raw,
            ai_response_raw::string AS ai_response,
            AI_COUNT_TOKENS('ai_complete', 'openai-gpt-5', user_prompt) AS input_tokens,
            AI_COUNT_TOKENS('ai_complete', 'openai-gpt-5', ai_response_raw::string) AS output_tokens
        FROM model_inference
    )
    SELECT 
        ai_response_raw,
        ai_response,
        input_tokens,
        output_tokens,
        (input_tokens / 1000000.0) * 0.75 AS input_credits,
        (output_tokens / 1000000.0) * 6.00 AS output_credits,
        (
            ((input_tokens / 1000000.0) * 0.75) + 
            ((output_tokens / 1000000.0) * 6.00)
        ) * 2.00 AS estimated_usd_cost
    FROM token_counting;
    """
    
    try:
        result_row = session.sql(ai_sql).collect()[0]
        end_time = time.time()
        exec_time = end_time - start_time
        
        input_tokens = result_row['INPUT_TOKENS']
        output_tokens = result_row['OUTPUT_TOKENS']
        
        input_cost = float(result_row['INPUT_CREDITS']) * 2.00
        output_cost = float(result_row['OUTPUT_CREDITS']) * 2.00
        total_cost = float(result_row['ESTIMATED_USD_COST'])
        
        try:
            parsed_gpt_resp = json.loads(result_row['AI_RESPONSE'])
            gpt_sections = parsed_gpt_resp.get("applicable_sections", [])
            print(f"✅ Found Applicable Sections: {gpt_sections}")
        except:
            gpt_sections = []
            print(f"✅ Raw GPT Output: {str(result_row['AI_RESPONSE'])[:200]}...")
        print(f"⏱️ Time Taken: {exec_time:.2f} seconds")
        print(f"📊 Input Tokens: {input_tokens:,} | Output Tokens: {output_tokens:,}")
        print(f"💵 GPT Total Cost: ${total_cost:.6f}")
        
        return {
            "Model": "OpenAI GPT-5 (Snowflake Cortex)",
            "Execution Time (s)": round(exec_time, 2),
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens,
            "Input Cost (USD)": input_cost,
            "Output Cost (USD)": output_cost,
            "Total Cost (USD)": total_cost,
            "Sections": gpt_sections
        }
        
    except Exception as e:
        print(f"Cortex SQL Error: {e}")
        return None

def run_judge_evaluation(session, contract_chunk: str, registry_data: dict, jev_sections: list, gpt_sections: list):
    print("\n" + "="*50)
    print("🧑‍⚖️ RUNNING AI JUDGE TO EVALUATE RESULTS")
    print("="*50)
    
    judge_prompt = f"""You are an impartial expert lease accounting judge.
Your task is to evaluate two AI models that extracted applicable accounting rules from a contract.

ACCOUNTING RULES REGISTRY:
{json.dumps(registry_data)}

CONTRACT TEXT:
{contract_chunk}

MODEL 1 (Jev) PREDICTED THESE APPLICABLE SECTIONS:
{jev_sections}

MODEL 2 (GPT) PREDICTED THESE APPLICABLE SECTIONS:
{gpt_sections}

INSTRUCTIONS:
1. Analyze the contract text and the rules.
2. Determine which sections actually apply based on the contract context.
3. Compare the predictions of Model 1 and Model 2.
4. Output your decision and reasoning.
"""
    safe_prompt = judge_prompt.replace("'", "''")
    
    schema_judge = {
        "type": "object",
        "properties": {
            "winner": {"type": "string", "enum": ["Jev", "GPT", "Tie"]},
            "reasoning": {"type": "string"}
        },
        "required": ["winner", "reasoning"],
        "additionalProperties": False
    }
    safe_schema = json.dumps(schema_judge).replace("'", "''")
    
    ai_sql = f"""
        SELECT SNOWFLAKE.CORTEX.AI_COMPLETE(
            model => 'openai-gpt-5',
            prompt => '{safe_prompt}',
            model_parameters => PARSE_JSON('{{"temperature": 0}}'),
            response_format => PARSE_JSON('{{"type": "json", "schema": {safe_schema}}}')
        ) AS judge_response
    """
    
    try:
        result_row = session.sql(ai_sql).collect()[0]
        parsed_judge_resp = json.loads(result_row['JUDGE_RESPONSE'])
        
        print("\n⚖️  THE JUDGE'S VERDICT:")
        print(f"🏆 Winner: {parsed_judge_resp.get('winner')}")
        print(f"💡 Reasoning:\n{parsed_judge_resp.get('reasoning')}")
        
    except Exception as e:
        print(f"Error running Judge: {e}")

def write_to_excel_csv(results: list, filename="benchmark_results.csv"):
    """Writes the gathered metrics into a CSV file."""
    if not results:
        print("No results to write.")
        return
        
    keys = results[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    print(f"\n📊 Benchmark completed! Results successfully saved to Excel/CSV: {os.path.abspath(filename)}")

# ==========================================
# RUN BLOCK
# ==========================================

def run_benchmark(session, pdf_filename: str):
    
    # Attempt to load the registry file. If it's missing on the other person's computer, fall back to a mock!
    registry_path = "Lease_assesmnt_registry.json"
    try:
        with open(registry_path, "r") as f:
            real_registry = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Warning: Could not find {registry_path}. Using a mock registry for the demo.")
        real_registry = {
            "1.1": {
                "title": "Scope Exceptions",
                "highlights": ["ASC 842 does not apply to intangible assets or inventory."],
                "sub_sections": {
                    "1.1.1": {
                        "title": "Service concession arrangements",
                        "highlights": ["Excluded from the scope of ASC 842."]
                    }
                }
            }
        }
    
    dummy_questions = {
        2: "Is there an identified asset?"
    }
    
    # Get Real Contract Data from Snowflake Stage
    chunks = get_real_ocr_chunks(session, pdf_filename)
    test_chunk = chunks[0]
    
    print(f"\nStarting Benchmark Comparison on a contract chunk of {len(test_chunk):,} characters...")
    
    # Run Benchmark Head-to-Head
    jev_metrics = evaluate_with_jev(test_chunk, real_registry, dummy_questions)
    gpt_metrics = evaluate_with_gpt_cortex(session, test_chunk, real_registry, dummy_questions)
    
    if jev_metrics and gpt_metrics:
        time_diff = gpt_metrics["Execution Time (s)"] - jev_metrics["Execution Time (s)"]
        print("\n" + "="*50)
        print("🏆 COMPARISON RESULTS")
        print("="*50)
        if time_diff > 0:
            print(f"🏎️ Jev was {time_diff:.2f} seconds FASTER than GPT!")
        elif time_diff < 0:
            print(f"🏎️ GPT was {abs(time_diff):.2f} seconds FASTER than Jev!")
        else:
            print("🏎️ Both models took the exact same time!")
            
        cost_diff = gpt_metrics["Total Cost (USD)"] - jev_metrics["Total Cost (USD)"]
        if cost_diff > 0:
            print(f"💰 Jev was ${cost_diff:.6f} CHEAPER than GPT!")
        elif cost_diff < 0:
            print(f"💰 GPT was ${abs(cost_diff):.6f} CHEAPER than Jev!")
            
        run_judge_evaluation(session, test_chunk, real_registry, jev_metrics.get("Sections", []), gpt_metrics.get("Sections", []))
    
    # Generate Excel / CSV Report
    metrics_list = []
    if jev_metrics: metrics_list.append(jev_metrics)
    if gpt_metrics: metrics_list.append(gpt_metrics)
    
    write_to_excel_csv(metrics_list, filename="benchmark_results.csv")

if __name__ == "__main__":
    # 1. Connect to Snowflake Dev Environment
    load_dotenv()
    token = os.getenv("Dev_TOKEN")
    if not token:
        print("⚠️ Environment variable 'DEV_TOKEN' is not set. You will need to put your password here to run it.")
        # Alternatively hardcode it for the person you share it with:
        # token = "YOUR_PASSWORD"

    connection_parameters = {
        "user":      "v-rreji",  # Change this to whoever is running it
        "password":  token,
        "account":   "HFSINCLAIR-DEV",
        "database":  "CONFORMED",
        "schema":    "HFS_FI_LEASE",
        "warehouse": "HFS_CUSTOM_WH",
        "role":      "DEVELOPER_FR",
    }

    print("🔌 Connecting to Snowflake Dev Environment...")
    session = Session.builder.configs(connection_parameters).create()

    # 2. Put your PDF filename here:
    my_pdf_filename = "4 - Ek-Stream TCP 23 Dec 2025.pdf"
    
    print(f"✅ Connected! Running benchmark on {my_pdf_filename}...")
    run_benchmark(session, my_pdf_filename)
