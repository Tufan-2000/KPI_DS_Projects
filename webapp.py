import streamlit as st
import os
import json
import sqlite3
import pandas as pd
import time
import re
from collections import defaultdict
from hashlib import md5
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

# LangChain imports (2025+)
from langchain_core.prompts import PromptTemplate
#from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


load_dotenv()

# Import all your existing functions here
# (I'll include the key ones, but you should import from your main module)

# ------------ Configuration ------------
EASY_MODEL = st.secrets("EASY_LLM", "gpt-3.5-turbo")
MEDIUM_MODEL = st.secrets("MEDIUM_LLM", "gpt-4o-mini")
HARD_MODEL = st.secrets("HARD_LLM", "gpt-4o")

_sql_cache = {}

# ------------ Helper Functions (Copy from your main code) ------------

def _hash_schema(schema_text: str) -> str:
    return md5(schema_text.encode("utf-8")).hexdigest()[:8]

def get_enhanced_spider_schema_sqlite(db_file_path):
    """Generate comprehensive schema"""
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema_parts = []
    table_relationships = defaultdict(list)
    
    for table in tables:
        table_info = []
        
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        foreign_keys = cursor.fetchall()
        
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
        
        if foreign_keys:
            fk_info = []
            for fk in foreign_keys:
                fk_info.append(f"  - {fk[3]} REFERENCES {fk[2]}({fk[4]})")
                table_relationships[table].append(f"{table}.{fk[3]} -> {fk[2]}.{fk[4]}")
            table_info.append("FOREIGN KEYS:\n" + "\n".join(fk_info))
        
        cursor.execute(f"SELECT * FROM {table} LIMIT 3")
        sample_rows = cursor.fetchall()
        
        if sample_rows:
            column_names = [col[1] for col in cols]
            sample_df = pd.DataFrame(sample_rows, columns=column_names)
            table_info.append(f"SAMPLE DATA (3 rows):\n{sample_df.to_string(index=False)}")
        
        schema_parts.append("\n".join(table_info))
    
    if table_relationships:
        rel_summary = ["TABLE RELATIONSHIPS:"]
        for table, relations in table_relationships.items():
            for rel in relations:
                rel_summary.append(f"  {rel}")
        schema_parts.append("\n".join(rel_summary))
    
    conn.close()
    return "\n" + "="*80 + "\n" + f"\n{'='*80}\n".join(schema_parts) + "\n" + "="*80

def execute_query_with_validation(db_file_path, query):
    """Execute query with error handling"""
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("EXPLAIN QUERY PLAN " + query)
        cursor.execute(query)
        result = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        
        try:
            return sorted(result) if result else [], column_names
        except TypeError:
            return result, column_names
            
    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}", []
    except Exception as e:
        return f"Execution Error: {str(e)}", []
    finally:
        conn.close()

def clean_sql_response(sql_text):
    """Clean SQL response"""
    lines = sql_text.split('\n')
    sql_lines = []
    
    for line in lines:
        line = line.strip()
        if (line and 
            not line.startswith('```') and 
            not line.startswith('#') and 
            not line.startswith('--') and
            not line.lower().startswith('explanation:') and
            not line.lower().startswith('answer:') and
            not line.lower().startswith('the sql')):
            sql_lines.append(line)
    
    return ' '.join(sql_lines)

def decompose_nl_question(question: str) -> dict:
    """Decompose natural language question"""
    q = question.strip().lower()
    
    wants_top_n = bool(re.search(r"\b(top|highest|most|lowest|least)\b\s+(\d+)", q))
    asks_count = "how many" in q or "count" in q
    asks_avg = "average" in q or "avg" in q
    asks_sum = "sum" in q or "total" in q
    asks_minmax = any(w in q for w in ["max","maximum","min","minimum","lowest","highest"])
    asks_distinct = any(w in q for w in ["unique","distinct","different"])
    has_group_by_cues = any(w in q for w in ["each","per","for each","by "]) or asks_avg or asks_count or asks_sum
    has_not_exists_cue = any(w in q for w in ["without","not","never","none","no "]) or "who haven't" in q
    has_compare_avg = "above average" in q or "higher than average" in q
    has_compare_all_any = "than all" in q or "than any" in q
    has_window_cue = any(w in q for w in ["rank","dense rank","row number","running","cumulative"])
    has_set_ops = any(w in q for w in ["except","intersect","union","either","or","both"])
    has_time_filter = any(w in q for w in ["before","after","between","during","since"]) or re.search(r"\b\d{4}\b", q)
    
    entity_like = re.findall(r"\b(customers?|orders?|products?|students?|courses?|songs?|singers?|teams?|matches?|players?|departments?|employees?|projects?)\b", q)
    needs_join = len(set(entity_like)) >= 2
    
    aggregations = []
    if asks_count: aggregations.append("COUNT")
    if asks_avg: aggregations.append("AVG")
    if asks_sum: aggregations.append("SUM")
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
        "primary_entities_guess": list(dict.fromkeys(entity_like)),
        "filters_detected": {
            "time_filter": has_time_filter,
            "negation": has_not_exists_cue,
            "compare_avg": has_compare_avg,
            "compare_all_any": has_compare_all_any,
        },
        "needs_join": needs_join,
        "needs_grouping": has_group_by_cues,
        "needs_window": has_window_cue or (wants_top_n and has_group_by_cues),
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
    """Classify query complexity"""
    hard_signals = [
        plan["needs_window"],
        plan["needs_set_ops"],
        plan["filters_detected"]["compare_avg"],
        plan["filters_detected"]["compare_all_any"],
        plan["filters_detected"]["negation"] and plan["needs_join"],
        plan["needs_join"] and plan["needs_grouping"] and plan["wants_top_n"],
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

def _build_routed_prompt(question: str, table_schema: str, plan: dict, previous_error: str = None, previous_sql: str = None) -> str:
    """Build prompt for SQL generation"""
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

=== QUESTION ===
{question}

=== DECOMPOSED PLAN ===
- Primary entities: {", ".join(plan.get("primary_entities_guess", []) or ["<unknown>"])}
- Joins needed: {plan["needs_join"]}
- Aggregations: {", ".join(plan["aggregations"]) if plan["aggregations"] else "None"}
- Grouping needed: {plan["needs_grouping"]}
- Window/RANK needed: {plan["needs_window"]}
- Distinct needed: {plan["needs_distinct"]}
- Ordering: {plan["ordering"] or "None"}
- Limit: {plan["limit"] or "None"}

=== SCHEMA ===
{table_schema}

{repair_block}

-- Return ONLY the SQL below:
"""

def _pick_llm_for_complexity(level: str) -> ChatOpenAI:
    """Select LLM based on complexity"""
    if level == "hard":
        return ChatOpenAI(model=HARD_MODEL, temperature=0)
    elif level == "medium":
        return ChatOpenAI(model=MEDIUM_MODEL, temperature=0)
    else:
        return ChatOpenAI(model=EASY_MODEL, temperature=0)

def create_sql_with_routed_pipeline(question: str, table_schema: str, previous_error: str = None, previous_sql: str = None):
    """Generate SQL from natural language"""
    schema_key = _hash_schema(table_schema)
    cache_key = f"{schema_key}::{question.strip().lower()}"
    if cache_key in _sql_cache and not previous_error:
        return _sql_cache[cache_key]
    
    plan = decompose_nl_question(question)
    level = classify_complexity_from_plan(plan)
    
    routed_prompt = _build_routed_prompt(question, table_schema, plan, previous_error, previous_sql)
    prompt_template = PromptTemplate.from_template("{prompt}")
    llm_chain = ChatOpenAI(
        llm=_pick_llm_for_complexity(level),
        prompt=prompt_template
    )
    response = llm_chain.invoke({"prompt": routed_prompt})
    sql = response["text"].strip()
    
    sql = clean_sql_response(sql)
    if not previous_error:
        _sql_cache[cache_key] = sql
    return sql

# ------------ Streamlit App ------------

def init_session_state():
    """Initialize session state variables"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'db_path' not in st.session_state:
        st.session_state.db_path = None
    if 'schema' not in st.session_state:
        st.session_state.schema = None
    if 'db_name' not in st.session_state:
        st.session_state.db_name = None

def get_database_info(db_path):
    """Get tables and row counts from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    table_info = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        table_info[table] = count
    
    conn.close()
    return table_info

def main():
    st.set_page_config(
        page_title="Text-to-SQL Generator",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
        }
        .error-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Text-to-SQL")
        st.markdown("---")
        
        st.subheader("📁 Database Setup")
        
        # Database selection method
        db_method = st.radio(
            "Choose database source:",
            ["Upload SQLite file", "Enter file path"],
            help="Select how you want to provide the database"
        )
        
        if db_method == "Upload SQLite file":
            uploaded_file = st.file_uploader(
                "Upload .sqlite or .db file",
                type=['sqlite', 'db', 'sqlite3'],
                help="Upload your SQLite database file"
            )
            
            if uploaded_file:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.session_state.db_path = temp_path
                st.session_state.db_name = uploaded_file.name
                st.success(f"✅ Loaded: {uploaded_file.name}")
        
        else:
            db_path_input = st.text_input(
                "Database file path:",
                placeholder="e.g., /path/to/database.sqlite",
                help="Enter the full path to your SQLite database"
            )
            
            if db_path_input and os.path.exists(db_path_input):
                st.session_state.db_path = db_path_input
                st.session_state.db_name = os.path.basename(db_path_input)
                st.success(f"✅ Loaded: {os.path.basename(db_path_input)}")
            elif db_path_input:
                st.error("❌ File not found!")
        
        # Load schema button
        if st.session_state.db_path:
            if st.button("🔄 Load Schema", use_container_width=True):
                with st.spinner("Loading schema..."):
                    try:
                        st.session_state.schema = get_enhanced_spider_schema_sqlite(st.session_state.db_path)
                        st.success("Schema loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading schema: {str(e)}")
        
        st.markdown("---")
        
        # Database info
        if st.session_state.db_path and st.session_state.schema:
            st.subheader("📊 Database Info")
            try:
                table_info = get_database_info(st.session_state.db_path)
                
                for table, count in table_info.items():
                    st.text(f"📋 {table}: {count:,} rows")
                
                # Show schema in expander
                with st.expander("🔍 View Full Schema"):
                    st.code(st.session_state.schema, language="sql")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # Model settings
        with st.expander("⚙️ Model Settings"):
            st.text_input("Easy Model", value=EASY_MODEL, disabled=True)
            st.text_input("Medium Model", value=MEDIUM_MODEL, disabled=True)
            st.text_input("Hard Model", value=HARD_MODEL, disabled=True)
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    # Main content
    st.title("💬 Natural Language to SQL")
    st.markdown("Ask questions about your database in plain English!")
    
    # Check if database is loaded
    if not st.session_state.db_path or not st.session_state.schema:
        st.info("👈 Please load a database from the sidebar to get started")
        
        # Show example questions
        st.subheader("📝 Example Questions")
        st.markdown("""
        - How many customers are there?
        - Show me the top 5 products by sales
        - What is the average order value by month?
        - List all employees in the Sales department
        - Which products have never been ordered?
        """)
        return
    
    # Query input
    st.subheader(f"🗂️ Database: {st.session_state.db_name}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What are the top 10 customers by total order value?",
            height=100,
            key="question_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("🚀 Generate SQL", use_container_width=True, type="primary")
        
        # Auto-retry toggle
        auto_retry = st.checkbox("Auto-retry on error", value=True)
    
    # Generate SQL
    if generate_btn and question:
        with st.spinner("🤖 Generating SQL..."):
            try:
                start_time = time.time()
                
                # Generate SQL
                predicted_sql = create_sql_with_routed_pipeline(question, st.session_state.schema)
                
                # Get complexity
                plan = decompose_nl_question(question)
                complexity = classify_complexity_from_plan(plan)
                
                # Execute with retry
                max_retries = 2 if auto_retry else 0
                retry_count = 0
                success = False
                
                while retry_count <= max_retries and not success:
                    result, column_names = execute_query_with_validation(st.session_state.db_path, predicted_sql)
                    
                    if isinstance(result, str) and "Error" in result:
                        retry_count += 1
                        if retry_count <= max_retries:
                            st.warning(f"⚠️ Attempt {retry_count} failed, retrying...")
                            predicted_sql = create_sql_with_routed_pipeline(
                                question,
                                st.session_state.schema,
                                previous_error=result,
                                previous_sql=predicted_sql
                            )
                    else:
                        success = True
                
                execution_time = time.time() - start_time
                
                # Store in history
                history_entry = {
                    'question': question,
                    'sql': predicted_sql,
                    'result': result,
                    'column_names': column_names,
                    'complexity': complexity,
                    'success': success,
                    'execution_time': execution_time,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.history.insert(0, history_entry)
                
                # Display results
                st.markdown("---")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Complexity", complexity.upper())
                with col2:
                    st.metric("Status", "✅ Success" if success else "❌ Failed")
                with col3:
                    st.metric("Execution Time", f"{execution_time:.2f}s")
                with col4:
                    if success and not isinstance(result, str):
                        st.metric("Rows Returned", len(result))
                
                # SQL Query
                st.subheader("📝 Generated SQL")
                st.code(predicted_sql, language="sql")
                
                # Copy button
                if st.button("📋 Copy SQL"):
                    st.code(predicted_sql, language="sql")
                    st.success("SQL copied to clipboard!")
                
                # Results
                if success and not isinstance(result, str):
                    st.subheader("📊 Query Results")
                    
                    if result:
                        # Convert to DataFrame
                        df = pd.DataFrame(result, columns=column_names)
                        
                        # Display table
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            csv,
                            "query_results.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        # Visualization options for numeric data
                        if len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                            st.subheader("📈 Visualization")
                            
                            viz_type = st.selectbox(
                                "Chart type:",
                                ["Bar Chart", "Line Chart", "Pie Chart"]
                            )
                            
                            if viz_type == "Bar Chart":
                                fig = px.bar(df, x=df.columns[0], y=df.columns[1])
                            elif viz_type == "Line Chart":
                                fig = px.line(df, x=df.columns[0], y=df.columns[1])
                            else:
                                fig = px.pie(df, names=df.columns[0], values=df.columns[1])
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Query executed successfully but returned no results.")
                
                else:
                    st.error(f"❌ Error: {result}")
            
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    # Query History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Query History")
        
        for idx, entry in enumerate(st.session_state.history[:10]):  # Show last 10
            with st.expander(f"{'✅' if entry['success'] else '❌'} {entry['question'][:80]}... ({entry['timestamp']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Question:** {entry['question']}")
                    st.code(entry['sql'], language="sql")
                
                with col2:
                    st.metric("Complexity", entry['complexity'].upper())
                    st.metric("Time", f"{entry['execution_time']:.2f}s")
                    
                    if st.button(f"🔄 Rerun", key=f"rerun_{idx}"):
                        st.session_state.question_input = entry['question']
                        st.rerun()
                
                if entry['success'] and not isinstance(entry['result'], str) and entry['result']:
                    df = pd.DataFrame(entry['result'], columns=entry['column_names'])
                    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":

    main()




