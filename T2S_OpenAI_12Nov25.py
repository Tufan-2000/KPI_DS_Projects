import os
import json
import sqlite3
import pandas as pd
import time
import re
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from collections import defaultdict
from functools import lru_cache
from hashlib import md5
from dotenv import load_dotenv
load_dotenv()

# ------------ Enhanced Schema Generation with Relationships and Statistics ------------

def get_enhanced_spider_schema_sqlite(db_file_path):
    """
    Generate comprehensive schema with relationships, constraints, and statistical information
    for better LLM understanding and SQL generation accuracy.
    """
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema_parts = []
    table_relationships = defaultdict(list)
    
    # Build comprehensive schema information
    for table in tables:
        table_info = []
        
        # Get column information with detailed metadata
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        foreign_keys = cursor.fetchall()
        
        # Build CREATE TABLE statement with constraints
        col_defs = []
        for col in cols:
            col_name, col_type, not_null, default_val, pk = col[1], col[2], col[3], col[4], col[5]
            col_def = f"  {col_name} {col_type}"
            if pk:
                col_def += " PRIMARY KEY"
            if not_null:
                col_def += " NOT NULL"
            if default_val:
                col_def += f" DEFAULT {default_val}"
            col_defs.append(col_def)
        
        create_stmt = f"CREATE TABLE {table} (\n" + ",\n".join(col_defs) + "\n);"
        table_info.append(create_stmt)
        
        # Add foreign key relationships
        if foreign_keys:
            fk_info = []
            for fk in foreign_keys:
                fk_info.append(f"  - {fk[3]} REFERENCES {fk[2]}({fk[4]})")
                table_relationships[table].append(f"{table}.{fk[3]} -> {fk[2]}.{fk[4]}")
            table_info.append("FOREIGN KEYS:\n" + "\n".join(fk_info))
        
        # Get sample data with value statistics
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        sample_rows = cursor.fetchall()
        
        if sample_rows:
            column_names = [col[1] for col in cols]
            sample_df = pd.DataFrame(sample_rows, columns=column_names)
            table_info.append(f"SAMPLE DATA (5 rows):\n{sample_df.to_string(index=False)}")
            
            # Add column statistics and unique values for categorical columns
            stats_info = []
            for col_name, col_type in [(col[1], col[2]) for col in cols]:
                try:
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    total_rows = cursor.fetchone()[0]
                    
                    # Get distinct values for potential categorical columns
                    cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table}")
                    distinct_count = cursor.fetchone()[0]
                    
                    # If it's likely categorical (low cardinality), show all values
                    if distinct_count <= 20 and distinct_count > 1:
                        cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} ORDER BY {col_name}")
                        unique_vals = [str(row[0]) for row in cursor.fetchall()]
                        stats_info.append(f"  {col_name}: DISTINCT VALUES = {unique_vals}")
                    
                    # For numeric columns, show range
                    elif col_type.upper() in ['INTEGER', 'REAL', 'NUMERIC']:
                        cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}) FROM {table} WHERE {col_name} IS NOT NULL")
                        min_max = cursor.fetchone()
                        if min_max and min_max[0] is not None:
                            stats_info.append(f"  {col_name}: RANGE = {min_max[0]} to {min_max[1]}")
                    
                except Exception:
                    continue
            
            if stats_info:
                table_info.append("COLUMN STATISTICS:\n" + "\n".join(stats_info))
        
        schema_parts.append("\n".join(table_info))
    
    # Add relationship summary
    if table_relationships:
        rel_summary = ["TABLE RELATIONSHIPS:"]
        for table, relations in table_relationships.items():
            for rel in relations:
                rel_summary.append(f"  {rel}")
        schema_parts.append("\n".join(rel_summary))
    
    return "\n" + "="*80 + "\n" + f"\n{'='*80}\n".join(schema_parts) + "\n" + "="*80

# ------------ Enhanced Query Execution with Better Error Handling ------------
def execute_query_with_validation(db_file_path, query):
    """
    Execute query with enhanced error handling and result validation
    """
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    try:
        # Validate query syntax first
        cursor.execute("EXPLAIN QUERY PLAN " + query)
        
        # Execute the actual query
        cursor.execute(query)
        result = cursor.fetchall()
        
        # Sort results for consistent comparison
        try:
            return sorted(result) if result else []
        except TypeError:
            # If sorting fails due to mixed types, return as-is
            return result
            
    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Execution Error: {str(e)}"
    finally:
        conn.close()

def clean_sql_response(sql_text):
    """
    Clean the SQL response to remove any extra text, comments, or formatting
    """
    lines = sql_text.split('\n')
    sql_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, markdown, or explanatory text
        if (line and 
            not line.startswith('```') and 
            not line.startswith('#') and 
            not line.startswith('--') and
            not line.lower().startswith('explanation:') and
            not line.lower().startswith('answer:') and
            not line.lower().startswith('the sql')):
            sql_lines.append(line)
    
    return ' '.join(sql_lines)

# -------------------- NL Question Decomposer + Router (ADD) --------------------

# You can override these via env vars if you like
EASY_MODEL   = os.getenv("EASY_LLM",   "gpt-3.5-turbo")
MEDIUM_MODEL = os.getenv("MEDIUM_LLM", "gpt-4o-mini")
HARD_MODEL   = os.getenv("HARD_LLM",   "gpt-4o")  # stronger model for hard cases

# very small in-memory cache to cut repeat calls
_sql_cache = {}

def _hash_schema(schema_text: str) -> str:
    # short schema hash to avoid massive cache keys
    return md5(schema_text.encode("utf-8")).hexdigest()[:8]

def decompose_nl_question(question: str) -> dict:
    """
    Lightweight, deterministic NL decomposition (no LLM needed).
    Extracts intent signals the SQL generator can use.
    """
    q = question.strip().lower()
    tokens = re.findall(r"[a-zA-Z_]+", q)

    # intent words
    agg_words = {"sum","total","average","avg","count","maximum","minimum","max","min","median","std","variance"}
    join_words = {"with","their","its","and","by","for","from","of"}  # weak signal, refined below
    distinct_words = {"unique","distinct","different"}
    having_words = {"more than","greater than","less than","fewer than","at least","at most"}
    not_words = {"without","not","never","none","no "}
    window_words = {"rank","dense rank","row number","running total","cumulative","percentile","ntile","top per"}
    setop_words = {"except","intersect","union","either","or","both"}
    time_words = {"before","after","between","during","on","since","until"}

    # features
    wants_top_n = bool(re.search(r"\b(top|highest|most|lowest|least)\b\s+(\d+)", q))
    asks_count  = "how many" in q or "count" in q
    asks_avg    = "average" in q or "avg" in q
    asks_sum    = "sum" in q or "total" in q
    asks_minmax = any(w in q for w in ["max","maximum","min","minimum","lowest","highest"])
    asks_distinct = any(w in q for w in distinct_words)
    has_group_by_cues = any(w in q for w in ["each","per","for each","by "]) or asks_avg or asks_count or asks_sum
    has_not_exists_cue = any(w in q for w in not_words) or "who haven't" in q or "who have not" in q
    has_compare_avg = "above average" in q or "higher than average" in q or "greater than average" in q
    has_compare_all_any = "than all" in q or "than any" in q
    has_window_cue = any(w in q for w in window_words) or re.search(r"\b(rank|row number)\b", q)
    has_set_ops = any(w in q for w in setop_words)
    has_time_filter = any(w in q for w in time_words) or re.search(r"\b\d{4}\b", q)

    # rough join signal: mentions of 2+ entity nouns separated by preps like "with/by/of"
    # this is heuristic but helps avoid an extra LLM pass
    entity_like = re.findall(r"\b(customers?|orders?|products?|students?|courses?|songs?|singers?|teams?|matches?|players?|departments?|employees?|projects?)\b", q)
    needs_join = len(set(entity_like)) >= 2

    aggregations = []
    if asks_count: aggregations.append("COUNT")
    if asks_avg:   aggregations.append("AVG")
    if asks_sum:   aggregations.append("SUM")
    if asks_minmax: aggregations.extend(["MIN/MAX"])

    ordering = None
    order_match = re.search(r"\b(oldest|youngest|highest|lowest|most|least|top|bottom)\b", q)
    if order_match:
        ordering = order_match.group(0)

    limit = None
    lim = re.search(r"\b(top|bottom)\s+(\d+)", q)
    if lim:
        limit = int(lim.group(2))

    plan = {
        "primary_entities_guess": list(dict.fromkeys(entity_like)),  # keep order, dedup
        "asks_fields_guess": None,     # will be decided by LLM using schema
        "filters_detected": {
            "time_filter": has_time_filter,
            "negation": has_not_exists_cue,
            "compare_avg": has_compare_avg,
            "compare_all_any": has_compare_all_any,
        },
        "needs_join": needs_join,
        "needs_grouping": has_group_by_cues,
        "needs_window": has_window_cue or (wants_top_n and has_group_by_cues),  # top N per group ⇒ window likely
        "needs_set_ops": has_set_ops,
        "needs_distinct": asks_distinct,
        "aggregations": aggregations,
        "ordering": ordering,
        "limit": limit,
        "wants_top_n": wants_top_n,
        "raw_question": question,
    }
    return plan

def classify_complexity_from_plan(plan: dict) -> str:
    """
    Classify EASY / MEDIUM / HARD purely from the NL plan.
    """
    hard_signals = [
        plan["needs_window"],
        plan["needs_set_ops"],
        plan["filters_detected"]["compare_avg"],
        plan["filters_detected"]["compare_all_any"],
        plan["filters_detected"]["negation"] and plan["needs_join"],  # NOT + join → NOT EXISTS/LEFT JOIN IS NULL
        plan["needs_join"] and plan["needs_grouping"] and plan["wants_top_n"],  # top-N per group
    ]
    if any(hard_signals):
        return "hard"

    medium_signals = [
        plan["needs_join"],
        plan["needs_grouping"],
        bool(plan["aggregations"]),
        plan["needs_distinct"],
        plan["ordering"] is not None,
    ]
    if any(medium_signals):
        return "medium"

    return "easy"

def _build_routed_prompt(question: str, table_schema: str, plan: dict, previous_error: str | None = None, previous_sql: str | None = None) -> str:
    """
    Short targeted prompt that leverages your original long prompt content implicitly.
    It keeps your strict rules but injects the structured plan to guide the SQL.
    """
    repair_block = ""
    if previous_error:
        repair_block = f"""

=== PREVIOUS ATTEMPT & ERROR (for correction) ===
SQL Candidate:
{previous_sql}

Error:
{previous_error}

Please correct the SQL. Keep to SQLite syntax and the rules below.
"""

    return f"""You are an expert SQLite SQL generator. Your task: OUTPUT ONLY the SQL query that answers the question. No explanations, no extra text.

Your process should be:
1.  **Understand the Goal:** Deconstruct the natural language question to identify the main entities, requested information, filters, and required relationships (joins, subqueries).
2.  **Map to Schema:** Match every identified entity and field to a table and column name in the provided schema. Note any required joins.
3.  **Construct the Query:** Build the query step-by-step, starting with the main table and adding joins, WHERE, GROUP BY, HAVING, ORDER BY, and LIMIT clauses as needed.

=== INSTRUCTIONS ===
1. STRICTLY use only tables & columns from the schema.
2. NEVER invent names, values, or relationships not in schema.
3. Use proper JOINs based on foreign keys.
4. Use WHERE for filters, HAVING for aggregated filters.
5. GROUP BY all non-aggregated SELECT columns.
6. Use DISTINCT if uniqueness is implied.
7. Use ORDER BY + LIMIT for ranking/top-N queries.
8. Always match exact case-sensitive column/table names.
9. Handle NULL with IS NULL / IS NOT NULL (never = NULL).
10. For text → single quotes; for dates → DATE('YYYY-MM-DD').

=== QUICK LOGIC STEPS ===
• Identify main entity.  
• Identify requested info (fields, counts, totals, etc.).  
• Identify filters, joins, aggregations.  
• Construct SQL following rules above.  
• Validate against schema before final output.  


=== QUESTION (Natural Language) ===
{question}

=== DECOMPOSED PLAN (use this faithfully) ===
- Primary entities (guess): {", ".join(plan.get("primary_entities_guess", []) or ["<unknown>"])}
- Joins needed: {plan["needs_join"]}
- Aggregations: {", ".join(plan["aggregations"]) if plan["aggregations"] else "None"}
- Grouping needed: {plan["needs_grouping"]}
- Window/RANK needed: {plan["needs_window"]}
- Set operations needed (UNION/INTERSECT/EXCEPT): {plan["needs_set_ops"]}
- Distinct needed: {plan["needs_distinct"]}
- Ordering cue: {plan["ordering"] or "None"}
- Limit (top/bottom N): {plan["limit"] or "None"}
- Filters: {plan["filters_detected"]}

=== SCHEMA (authoritative, obey exact names) ===
{table_schema}


{repair_block}

-- Return ONLY the SQL below (no comments, no prose):
"""

def _pick_llm_for_complexity(level: str) -> ChatOpenAI:
    """
    Tiered LLM selection. You can tighten or loosen models as you wish.
    """
    if level == "hard":
        return ChatOpenAI(model=HARD_MODEL, temperature=0)
    elif level == "medium":
        return ChatOpenAI(model=MEDIUM_MODEL, temperature=0)
    else:
        return ChatOpenAI(model=EASY_MODEL, temperature=0)

def create_sql_with_routed_pipeline(question: str, table_schema: str, previous_error: str | None = None, previous_sql: str | None = None):
    """
    New entry point:
    1) Decompose NL question (no API call)
    2) Classify complexity (easy/medium/hard)
    3) Route to LLM tier
    4) Cache results keyed by (question, schema-hash)
    """
    schema_key = _hash_schema(table_schema)
    cache_key = f"{schema_key}::{question.strip().lower()}"
    if cache_key in _sql_cache and not previous_error:
        return _sql_cache[cache_key]

    plan = decompose_nl_question(question)
    level = classify_complexity_from_plan(plan)

    # Build prompt
    routed_prompt = _build_routed_prompt(question, table_schema, plan, previous_error, previous_sql)
    prompt_template = PromptTemplate.from_template("{prompt}")
    llm_chain = LLMChain(
        llm=_pick_llm_for_complexity(level),
        prompt=prompt_template
    )
    response = llm_chain.invoke({"prompt": routed_prompt})
    sql = response["text"].strip()

    # Clean and cache
    sql = clean_sql_response(sql)
    if not previous_error:
        _sql_cache[cache_key] = sql
    return sql

# ------------ Enhanced Evaluation with Detailed Analytics ------------
def enhanced_evaluate(dev_sql_path, db_root_path):
    """
    Enhanced evaluation with detailed error analysis and performance metrics
    """
    pairs = parse_dev_sql(dev_sql_path)
    
    # Metrics tracking
    metrics = {
        'correct': 0,
        'total': 0,
        'syntax_errors': 0,
        'logic_errors': 0,
        'execution_errors': 0,
        'query_types': defaultdict(int),
        'errors_by_complexity': defaultdict(int)
    }
    
    detailed_results = []
    start_index = 0 # User-defined starting index (inclusive)
    end_index = 15 # User-defined ending index (exclusive)

    # Ensure the range is valid and within bounds
    pairs_range = pairs[start_index:end_index]

    for i, (question, gold_sql, db_name) in enumerate(pairs_range, start=start_index):
        if not db_name:
            continue
            
        # Find database file
        candidate_path = os.path.join(db_root_path, db_name, f"{db_name}.sqlite")
        if not os.path.exists(candidate_path):
            candidate_path = os.path.join(db_root_path, f"{db_name}.sqlite")
            
        if not os.path.exists(candidate_path):
            print(f"⚠️ Database file not found for {db_name}, skipping.")
            continue

        # Generate enhanced schema
        schema = get_enhanced_spider_schema_sqlite(candidate_path)
        
        # Generate SQL with enhanced prompting
        predicted_sql = create_sql_with_routed_pipeline(question, schema)
        
        # Execute both queries
        gold_result = execute_query_with_validation(candidate_path, gold_sql)

        # 🔁 Retry loop for predicted SQL execution
        max_retries = 2
        retry_count = 0
        pred_result = None
        success = False

        while retry_count < max_retries and not success:
            pred_result = execute_query_with_validation(candidate_path, predicted_sql)

            if isinstance(pred_result, str) and "Error" in pred_result:
                retry_count += 1
                print(f"⚠️ SQL execution failed (attempt {retry_count}): {pred_result}")

                # Ask LLM to refine/fix the query
                predicted_sql = create_sql_with_routed_pipeline(
                    question, schema, previous_error=pred_result if isinstance(pred_result, str) else None, previous_sql=predicted_sql
                )
            else:
                success = True

        if not success:
            pred_result = f"SQL Error after {max_retries} retries"

        # FIX: Analyze query complexity using the plan, not the question string
        plan = decompose_nl_question(question)
        complexity = classify_complexity_from_plan(plan)
        metrics['query_types'][complexity] += 1
        
        # Detailed logging for every questions to be shown
        print(f"\n{'='*80}")
        print(f"Question {i+1}: {question}")
        print(f"Database: {db_name}")
        print(f"Complexity: {complexity.upper()}")
        print(f"{'='*80}")
        print(f"Gold SQL: {gold_sql}")
        print(f"Predicted SQL: {predicted_sql}")
        print(f"Gold Result: {gold_result}")
        print(f"Predicted Result: {pred_result}")
        
        # Evaluate result
        result_status = evaluate_query_result(gold_result, pred_result, predicted_sql)
        
        if result_status == 'correct':
            print("✅ CORRECT")
            metrics['correct'] += 1
        elif result_status == 'syntax_error':
            print("❌ SYNTAX ERROR")
            metrics['syntax_errors'] += 1
            metrics['errors_by_complexity'][complexity] += 1
        elif result_status == 'logic_error':
            print("❌ LOGIC ERROR")
            metrics['logic_errors'] += 1
            metrics['errors_by_complexity'][complexity] += 1
        else:
            print("❌ EXECUTION ERROR")
            metrics['execution_errors'] += 1
            metrics['errors_by_complexity'][complexity] += 1
        
        metrics['total'] += 1
        
        # Store detailed result
        detailed_results.append({
            'question': question,
            'db_name': db_name,
            'complexity': complexity,
            'gold_sql': gold_sql,
            'predicted_sql': predicted_sql,
            'status': result_status,
            'gold_result': str(gold_result)[:200] if not isinstance(gold_result, str) else gold_result,
            'pred_result': str(pred_result)[:200] if not isinstance(pred_result, str) else pred_result
        })
        
        # Progress update
        accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
        print(f"Running Accuracy: {accuracy:.2%} ({metrics['correct']}/{metrics['total']})")
    
    # Final comprehensive report
    print_comprehensive_report(metrics, detailed_results)
    return metrics, detailed_results


def evaluate_query_result(gold_result, pred_result, predicted_sql):
    """Evaluate query execution result with enhanced comparison logic"""
    if isinstance(pred_result, str) and ("Error" in pred_result):
        if "SQL Error" in pred_result:
            return 'syntax_error'
        else:
            return 'execution_error'
    elif isinstance(gold_result, str) and ("Error" in gold_result):
        return 'gold_error'
    elif gold_result == pred_result:
        return 'correct'
    else:
        # Check if it's just a column ordering issue for GROUP BY queries
        if is_equivalent_result(gold_result, pred_result):
            return 'correct'
        else:
            return 'logic_error'

def is_equivalent_result(gold_result, pred_result):
    """
    Check if two query results are semantically equivalent despite column ordering differences
    """
    if not gold_result or not pred_result:
        return gold_result == pred_result
    
    # If different number of rows, definitely different
    if len(gold_result) != len(pred_result):
        return False
    
    # If same number of columns, check if it's just reordered
    if len(gold_result[0]) == len(pred_result[0]) == 2:
        # For 2-column results (common in GROUP BY), check if sets are equivalent
        gold_set = set(gold_result)
        pred_set = set(pred_result)
        
        # Direct comparison
        if gold_set == pred_set:
            return True
            
        # Check if columns are just swapped
        swapped_pred = set((row[1], row[0]) for row in pred_result)
        if gold_set == swapped_pred:
            return True
    
    # For other cases, convert to sets and compare (handles row ordering)
    try:
        gold_set = set(gold_result)
        pred_set = set(pred_result)
        return gold_set == pred_set
    except TypeError:
        # If results contain unhashable types, fall back to sorted comparison
        try:
            return sorted(gold_result) == sorted(pred_result)
        except:
            return False
    
    return False

def print_comprehensive_report(metrics, detailed_results):
    """Print comprehensive evaluation report"""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION REPORT")
    print(f"{'='*80}")
    
    if metrics['total'] > 0:
        accuracy = metrics['correct'] / metrics['total']
        print(f"OVERALL ACCURACY: {accuracy:.2%} ({metrics['correct']}/{metrics['total']})")
        print(f"Target Achievement: {'✅ ACHIEVED' if accuracy >= 0.90 else '❌ NEEDS IMPROVEMENT'}")
    
    print(f"\nERROR BREAKDOWN:")
    print(f"  Syntax Errors: {metrics['syntax_errors']}")
    print(f"  Logic Errors: {metrics['logic_errors']}")
    print(f"  Execution Errors: {metrics['execution_errors']}")
    
    print(f"\nPERFORMANCE BY COMPLEXITY:")
    for complexity in ['easy', 'medium', 'hard']:
        total = metrics['query_types'][complexity]
        errors = metrics['errors_by_complexity'][complexity]
        if total > 0:
            success_rate = (total - errors) / total
            print(f"  {complexity.capitalize()}: {success_rate:.2%} ({total - errors}/{total})")
    
    print(f"\n{'='*80}")
    # Categorize errors from the list
    actual_errors = [
        err for err in detailed_results 
        if err.get("status") in ("logic_error", "syntax_error", "execution_error")
    ]

    if actual_errors:
        print("\nErrors encountered:")
        for err in actual_errors:
            print(json.dumps(err, indent=2))

# ------------ Parse dev.sql (unchanged) ------------
def parse_dev_sql(filepath):
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("Question"):
            question_part = lines[i].split(":", 1)[1].strip()
            if "|||" in question_part:
                question_text, db_name = map(str.strip, question_part.split("|||", 1))
            else:
                question_text, db_name = question_part, None

            if i + 1 < len(lines) and lines[i + 1].startswith("SQL:"):
                sql_query = lines[i + 1].split(":", 1)[1].strip()
            else:
                sql_query = ""
            pairs.append((question_text, sql_query, db_name))
            i += 2
        else:
            i += 1
    return pairs

# ------------ Main execution ------------
if __name__ == "__main__":
    DEV_SQL_PATH = r"C:\Users\TufanPaul\OneDrive - KPI PARTNERS INC\Desktop\CODING\ProjectPro\Tufan_Projectpro\dev.sql"
    DB_SQLITE_PATH = r"C:\Users\TufanPaul\OneDrive - KPI PARTNERS INC\Desktop\CODING\ProjectPro\T2S_OpenAI_Github_Website\spider_data\database"
    
    print("🚀 Starting Enhanced Text-to-SQL Evaluation")
    print("Target: 90%+ Accuracy")
    print("="*80)
    
    metrics, results = enhanced_evaluate(DEV_SQL_PATH, DB_SQLITE_PATH)
