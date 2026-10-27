'''AI Agents Core Mechanics: Structured Retrieval: Database Queries (SQL)

This allows relational databases to be exposed to langauge models. The language model acts as translator
natural langauge to SQl query, executes the query, record the data returned and adds it to the model context.

Model Requirements: Schema of the database, by including the metadata in the model's prompt instructions, examples of input and output

The following code initialises an in memory database representing customer accounts and exposes a querytool. 
The tool simulates a database level privilege engine by validating only SELECT statements are executed, raising permission errors for 
write operations. 
'''

import json
import asyncio
import sqlite3
from anthropic import AsyncAnthropic

# 1. Initialise database with customer records
db = sqlite3.connect(":memory:")
db.executescript(
    "CREATE TABLE customers (id INT, name TEXT, balance REAL, country TEXT);"
    "INSERT INTO customers VALUES (1, 'Alice Smith', 1250.50, 'US');"
    "INSERT INTO customers VALUES (2, 'Bob Jones', 8200.00, 'CA');"
    "INSERT INTO customers VALUES (3, 'Charlie Brown', 450.00, 'US');"
    "INSERT INTO customers VALUES (4, 'Diana Prince', 15000.00, 'UK');"
)

# 2. Expose a database query tool with security validation 
def run_readonly_query(sql_query):
    """
    Runs a single read only SQL query and returns the data in json dumps. 

    Args:
        sql_query: SQL command created by the LLM to execute. LLM acts as a Text-to-SQL translator.
    """
    print(f"    [EXECUTE] Running run_readonly_query on SQL datbase. Query: {sql_query}")
    clean_query = sql_query.strip().upper()

    if not clean_query.startswith("SELECT"):
        raise PermissionError("Database Error: Write privileges denied. Connection is READ-ONLY.")
    if ";" in clean_query and clean_query.count("SELECT") < clean_query.count(";"):
        raise PermissionError("Database Error: Multiple queries blocked. Only single SELECT statements allowed.")

    cursor = db.cursor()
    cursor.execute(sql_query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    results = [dict(zip(columns, row)) for row in rows]
    return json.dumps(results)

# 3. Register the tool schema 

tools = [{
        "name": "run_readonly_query",
        "description": "Runs a single read only SQL query and returns the data in json dumps. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql_query": {"type": "string", "description": "The exact SQL SELECT statement to run. Columns: id (int), name (text), balance (real), country (text)."},
                },
            "required": ["sql_query"]
            }
        }]


# 4. Agent Execution. Ask, run tool, return response, feedback to model. 

MODEL = "claude-opus-5-5"
client = AsyncAnthropic()
SYSTEM_PROMPT = (
    "You are a database access assistant. Translate user questions to SQL queries and use run_readonly_query to run them. Table schema: customers(id, name, balance, country)."
)

async def run_sql_agent(question):
    print(f"--- Starting Agent Loop for input: '{question}'---\n")

    # 1) Read the initial user question and store it in messages list for short term context memory
    messages = [
        {"role": "user", "content": question}
    ]

    max_turns = 5 
    for turn in range(max_turns):
        # Turn 1: Query the model with the tools registered
        resp = await client.messages.create(
                        model = MODEL,
                        system = SYSTEM_PROMPT,
                        messages=messages, # Query the language model with the context
                        tools=tools,
                        max_tokens=15000
                    )
        # Append the model response to messages
        messages.append({"role": "assistant", "content": resp.content})

        # resp.content is a list of blocks - typical reply looks like:
        '''
        [
            TextBlock(type="text", text="I'll calculate that for you."),
            ToolUseBlock(type="tool_use", id="toolu_1", name="calculate_loan_payment",
            input={"principal": 15000, "rate": 0.055, "years": 4}),
        ]
        '''

        # Print any text Claude wrote this turn
        for block in resp.content:
            if block.type == "text":
                print(block.text)

        # Done if Claude answered instead of asking for a tool
        if resp.stop_reason != "tool_use":
            break

        # 3) Act - run each tool Clause requests 
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_args = block.input
            print(f"[TURN {turn + 1}] Model requested tool: {tool_name} with args: {tool_args}")

            if tool_name == "run_readonly_query":
                try:
                    outcome = run_readonly_query(sql_query=tool_args["sql_query"])
                    is_error = False
                except Exception as e:  # errors go back to Claude so it can fix its query
                    outcome = f"Exception: {e}"
                    is_error = True
            else:
                outcome = f"Error. Unknown tool {tool_name}"
                is_error = True

            print(f"    [RESULT] {outcome}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": outcome,
                "is_error": is_error
            })

        # 4) Record - feed the results back to the context, then return to step 1
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"Stopped: reached {max_turns} turns before Claude finished.")

    return messages
            
        
query = "Who is our highest-balance customer in the US, and what is their balance?"
if __name__ == "__main__":
    asyncio.run(run_sql_agent(query))      