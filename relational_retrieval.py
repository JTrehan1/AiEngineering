'''AI Agents Core Mechanics: Structured Retrieval: Database Queries (SQL)

This allows relational databases to be exposed to langauge models. The language model acts as translator
natural langauge to SQl query, executes the query, record the data returned and adds it to the model context.

Model Requirements: Schema of the database, by including the metadata in the model's prompt instructions, examples of input and output

The following code initialises an in memory database representing customer accounts and exposes a querytool. 
The tool simulates a database level privilege engine by validating only SELECT statements are executed, raising permission errors for 
write operations. 
'''

import sqlite3
import json
from anthropic import AsyncAnthropic

# 1. Initialize database with customer records
db = sqlite3.connect(":memory:")
db.executescript(
    "CREATE TABLE customers (id INT, name TEXT, balance REAL, country TEXT);"
    "INSERT INTO customers VALUES (1, 'Alice Smith', 1250.50, 'US');"
    "INSERT INTO customers VALUES (2, 'Bob Jones', 8200.00, 'CA');"
    "INSERT INTO customers VALUES (3, 'Charlie Brown', 450.00, 'US');"
    "INSERT INTO customers VALUES (4, 'Diana Prince', 15000.00, 'UK');"
)

# 2. Expose a database query tool with securuty validation 
def run_readonly_query(sql_query):
    """
    Runs a read only SQL query.

    Args:
        sql_query: SQL command formed by the LLM to execute. 
    """
    print(f"    [EXECUTE] Running run_readonly_query on SQL datbase. Query: {sql_query}")
    clean_query = sql_query.strip().upper()

    if not clean_query.startswith("SELECT"):
        raise PermissionError("Database Error: Write privileges denied. Connection is READ-ONLY.")
    if ";" in clean_query and clean_query.count("SELECT") < clean_query.couint(";"):
        raise PermissionError("Database Error: Multiple queries blocked. Only single SELECT statements allowed.")

    cursor = db.cursor()
    cursor.execute(sql_query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    results = [dict(zip(columns, row)) for row in rows]
    return json.dumps(results)