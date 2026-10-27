'''AI Agents Core Mechanics: Coordinating Multiple Tools

Agent systems present multiple tool schemas to the LLM where the LLM acts as a dynamic router and coordinator.

The agent is responsible for:

1) Intent Classification: The model evaluates natural language input and determines which schema is semantically aligned with the task.

2) Parameter Extraction: Extracts parameters specific to the selected schema.

3) Sequential Coordination: Model chains tools in sequence. Outputs of one tool are parameters for the next.

The following provides the agent with access to both the SQL database tool and unstructured search index. 
When given a compound request the model executes a multi-turn sequence: First query the SQL database, 
then search the policy manual and finally combine the result.
'''

import asyncio
import json
import sqlite3
from anthropic import AsyncAnthropic

# 1. Structured Database setup
db = sqlite3.connect(":memory:")
db.executescript(
    "CREATE TABLE customers (id INT, name TEXT, balance REAL, country TEXT);"
    "INSERT INTO customers VALUES (1, 'Alice Smith', 1250.50, 'US');"
    "INSERT INTO customers VALUES (2, 'Bob Jones', 8200.00, 'CA');"
)

# 2. Unstructured Policy setup
policies = [
    {"id": 1, "text": "General Returns: Refunds are allowed within 30 days of purchase. The item must be unused, in its original packaging, and accompanied by a receipt."},
    {"id": 2, "text": "Opened Items: If an item has been opened or used, it is eligible for exchange or store credit only, provided it is returned within 14 days of purchase."}
]

# 3. Tool 1: SQL Database Query
def run_readonly_query(sql_query):
    """
    Runs a single read only SQL query and returns the data in json dumps. 

    Args:
        sql_query: SQL command created by the LLM to execute. LLM acts as a Text-to-SQL translator.
    """
    print(f"    [EXECUTE] Running run_readonly_query on SQL database. Query: {sql_query}")
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

# 4. Tool 2: RAG Policy search
def search_policies(query):
    print(f"  [EXECUTE] Scanning policy store for query terms: '{query}'...")
    # Simple keyword-overlap stand-in for semantic vector search
    query_tokens = set(query.lower().replace("?", "").replace(".", "").split())
    matches = [] 

    for policy in policies:
        # Calculate the token intersection score
        policy_tokens = set(policy["text"].lower().replace(".","?").replace(",", "").split())
        overlap = len(query_tokens.intersection(policy_tokens))
        matches.append((overlap, policy))

    # Sort by matching score in descending order
    matches.sort(key=lambda x: x[0], reverse=True)
    results = [match[1] for match in matches[:2] if match[0]>0]

    if not results:
        return "No matching policy terms found"

    return json.dumps(results)

# 5. Register both schemas under tools
tools = [
    {
        "name": "run_readonly_query",
        "description": "Runs a single read only SQL query and returns the data in json dumps. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql_query": {"type": "string", "description": "Execute a raw SQL query on the database. Columns: id (int), name (text), balance (real), country (text). Only SELECT statements permitted."},
                },
            "required": ["sql_query"]
            }
    },
    {
            "name": "search_policies",
            "description": "Search the corporate policy database for return rules and guidelines.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Key terms to search for (e.g. 'opened electronics', 'refund days')"},
                },
                "required": ["query"]
            }
    }
]

# 6. Agent Execution. Ask, run tool, return response, feedback to model. 
MODEL = "claude-opus-5-5"
client = AsyncAnthropic()
SYSTEM_PROMPT = (
    "You are a customer support assistant. Use run_readonly_query to search customer records and search_policies to search corporate returns policy. "
)
async def run_multi_tool_agent(question):
    print(f"--- Starting Agent Loop for input: '{question}'---\n")

    # 1. Read the initial user question and store it in messages list for short term context memory
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

        # 3. Act - run each tool Claude requests 
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
                except Exception as e:
                    outcome = f"Exception: {e}"
                    is_error = True

            elif tool_name == "search_policies":
                try:
                    outcome = search_policies(query=tool_args["query"])
                    is_error = False
                except Exception as e:
                    outcome = f"Exception {e}"
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

        # 4. Record - feed the results back to the context, then return to step 1
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"Stopped: reached {max_turns} turns before Claude finished.")  

    return messages         

query = "Check customer Bob Jones's current balance, and find our policy regarding returns of opened items."
if __name__ == "__main__":
    asyncio.run(run_multi_tool_agent(query)) 
    
