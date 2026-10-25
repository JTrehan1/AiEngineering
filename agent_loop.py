'''AI Agents: The Core Loop - Bare Metal Implementation

Execution pattern of an agent follows 4 phases
1) Observe - Read the initial user query and accumulated historical context of the conversation. Each time round, Claude re-reads all messages.
2) Decide - Query the language model with the context. Either return a conversational answer or a request to use a tool
3) Act - If a tool is needed, run tool with arguments and capture the output. 
4) Record - Append the model's request and the results back into the conversation history. Return to step 1.

This python file will implement a bare metal execution loop using an agent to determine the shipping status of an online order.
'''
import asyncio
import sqlite3
from anthropic import AsyncAnthropic

MODEL = "claude-opus-5-5"
client = AsyncAnthropic()
SYSTEM_PROMPT = (
    "You are a customer service assistant. "
    "Use the query_delivery_status tool to check order statuses for customers."
)

# Create an in memory database representing commercial order state
db = sqlite3.connect(":memory:")
db.executescript(
    "CREATE TABLE orders(id INT, status TEXT, date TEXT, carrier TEXT);"
    "INSERT INTO orders VALUES (982, 'Shipped', 'May 28', 'FedEx');"
    "INSERT INTO orders VALUES (104, 'Processing', 'Pending', 'None');"
)

# Define a local function to query the database 
def query_delivery_status(order_id):
    """
    Look up an order by ID and return its status, date and carrier.
    Args:
        order_id: Integer value representing an order_id.
    """
    # Debug line
    print(f"[EXECUTE] Running local database lookup for order {order_id}")
    cursor = db.cursor()
    cursor.execute("SELECT status, date, carrier FROM orders WHERE id = ?", (order_id,)) # Query the database
    row = cursor.fetchone() # Store the result from the sql query in row
    if row:
        return f"Status: {row[0]}, Date: {row[1]}, Carrier: {row[2]}"
    return "Order not found"

# Register the function schema signature
tools = [{
        "name": "query_delivery_status",
        "description": "Look up the status, shipment date and carrier from the order table based on an order id",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer", "description": "The order ID to check"}
                },
            "required": ["order_id"]
            }
        }]

# Execute the agentic feedback loop
async def run_agent(question):
    print(f"--- Starting Agent Loop for input: '{question}'---\n")

    # 1) Read the initial user query and store it in the messages list for short term context memory. 
    messages = [
        {"role": "user", "content": question}
    ]

    
    max_turns = 5
    for turn in range(max_turns):
        print(f"[TURN {turn + 1}] Querying model...")
        # 2) Decide, - Query the language model with the context. Return either a conversational reply or request to use a tool
        resp = await client.messages.create(
            model = "claude-opus-5-5",
            system = SYSTEM_PROMPT,
            messages=messages, # Query the language model with the context
            tools=tools,
            max_tokens=15000
        )

        # Append the reply. 4) Record
        msg = resp.content
        messages.append({"role": "assistant", "content": msg})

        # Print any text from Claude this turn
        for block in resp.content:
            if block.type == "text":
                print(block.text)

        # Terminate if the response is direct to the user
        if resp.stop_reason != "tool_use":
            print(f"[TURN {turn + 1}] Claude stopped because {resp.stop_reason}.")
            break

        # 3) Act if a tool is needed, run the tool with Claude's arguments and capture output
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            tool_name = block.name
            tool_args = block.input
            print(f"[TURN {turn + 1}] Model requested tool: {tool_name} with args: {tool_args}")

            if tool_name == "query_delivery_status":
                result = query_delivery_status(tool_args["order_id"])
                is_error = False
            else: 
                result = f"Unknown tool. {tool_name}"
                is_error = True

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id, # links this answer to Claude's request
                "content": result,
                "is_error": is_error,
            })
            print(f"[TURN {turn + 1}] Tool result recorded. Continuing loop...\n")

        # 4) Record
        messages.append({"role": "user", "content": tool_results})

    return messages

if __name__ == "__main__":
    asyncio.run(run_agent("Where is my order 982?"))