'''AI Agents Core Mechanics: Structured Retrieval: API Abstractions (REST)

Database backend is encapsulated; only the mock API client endpoints are registered as tools. The model must call these
endpoints. Better for security, query optimisation and handles modifications to table schema better. 

The following playground simulates an API Gateway. Rather than letting the model write raw SQL, 
expose two parameterized REST API endpoints:
'''

import json
import sqlite3
import asyncio
from anthropic import AsyncAnthropic

# Establish absolute completions endpoint proxy
url = (globals().get("_origin", "") or "") + "/api/agentic/model"
client = AsyncAnthropic(base_url=url)

# 1. Initialise backend database (fully encapsulated, hidden from LLM)
db = sqlite3.connect(":memory:")
db.executescript(
    "CREATE TABLE customers (id INT, name TEXT, email TEXT);"
    "CREATE TABLE orders (id INT, customer_id INT, amount REAL, status TEXT);"
    "INSERT INTO customers VALUES (1, 'Alice Smith', 'alice@example.com');"
    "INSERT INTO customers VALUES (2, 'Bob Jones', 'bob@example.com');"
    "INSERT INTO orders VALUES (101, 1, 150.00, 'Delivered');"
    "INSERT INTO orders VALUES (102, 1, 50.00, 'Shipped');"
    "INSERT INTO orders VALUES (103, 2, 85.50, 'Processing');"
)

# 2. REST API Implementations (Data Access Layer)
# These simulate standard GET endpoints: /api/customers/{id} and /api/orders/{id}

def get_customer_details(customer_id):
    print(f"  [GET /api/customers/{customer_id}] Fetching customer details...")
    cursor = db.cursor()
    # Parameterised query protects against SQL injection
    cursor.execute("SELECT id, name, email FROM customers WHERE id = ?", (customer_id,))
    row = cursor.fetchone()
    if row:
        return json.dumps({"id": row[0], "name": row[1], "email": row[2]})
    return json.dumps({"error": "Customer not found."})

def get_order_details(order_id):
    print(f"  [GET /api/orders/{order_id}] Fetching order details...")
    cursor = db.cursor()
    # Parameterised query protects against SQL injection
    cursor.execute("SELECT id, customer_id, amount, status FROM orders WHERE id = ?", (order_id,))
    row = cursor.fetchone()
    if row:
        return json.dumps({"id": row[0], "customer_id": row[1], "amount": row[2], "status": row[3]})
    return json.dumps({"error": "Order not found."})

# 3. Register the tool schema

tools = [
    {
        "name": "get_customer_details",
        "description": "Runs a SELECT SQL query and returns the customer id, name and email from customers table in json dumps. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer", "description": "The unique ID of the customer."},
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_order_details",
        "description": "Runs a SELECT SQL query and returns the customer id, amount and status from orders table in json dumps. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer", "description": "The unique ID of the order."},
            },
            "required": ["order_id"]
        }
    },
]

# 4. Agent Execution. Ask, run tool, return response, feedback to model.

MODEL = "claude-opus-5-5"
client = AsyncAnthropic()
SYSTEM_PROMPT = (
    "You are a database access assistant. Respond to user questions by calling get_customer_details and get_order_details. "
    "Never attempt to write raw database queries."
)

async def run_sql_agent(question):
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

        # 3. Act - run each tool Clause requests 
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_args = block.input
            print(f"[TURN {turn + 1}] Model requested tool: {tool_name} with args: {tool_args}")

            if tool_name == "get_customer_details":
                try:
                    outcome = get_customer_details(customer_id=tool_args["customer_id"])
                    is_error = False
                except Exception as e:
                    outcome = f"Exception: {e}"
                    is_error = True

            elif tool_name == "get_order_details":
                try:
                    outcome = get_order_details(order_id=tool_args["order_id"])
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

query = "Who placed order 101, and what is the status of that order?"
if __name__ == "__main__":
    asyncio.run(run_sql_agent(query)) 