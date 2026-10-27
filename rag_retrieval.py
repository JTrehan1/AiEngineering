'''AI Agents Core Mechanics: Unstructured Retrieval: RAG

Placing large unstructured data into a prompt is not practical, it uses too much of the model's context window. 

RAG. Retrieval Augmented Generation uses an external document store for relevant text segments and dynamically inserts only those segments into the model's prompt context.

Retrieval Pipeline:
Isolate and expose unstructured data.

1) Chunking: Source documents split into smaller, overlapping segments to preserve contextual boundaries.

2) Indexing: Chunks are passed through an embedding model to generate a high dimensional vector. 
    Vectors represent the semantic meaning of the text. Vectors are stored in a VectorDB.

3) Querying: Application converts the query into a vector and performs a similarity search against the VectorDB.

4) Generation: The top-K matching chunks are extracted and inserted into the system instructions as context. 
    The model can generate a fact based answer. 

The following implements local text search tool containing retail policy chunks. 
The model decides to search the policy store, parses the matching chunks and uses them to determine
the correct refund or store credit terms. 
'''
import json
import asyncio
from anthropic import AsyncAnthropic

# 1. Store policy document chunks

policies = [
    {
        "id": 1, 
        "text": "General Returns: Refunds are allowed within 30 days of purchase. The item must be unused, in its original packaging, and accompanied by a receipt."
    },
    {
        "id": 2, 
        "text": "Opened Items: If an item has been opened or used, it is eligible for exchange or store credit only, provided it is returned within 14 days of purchase."
    },
    {
        "id": 3, 
        "text": "Electronics Policy: Electronic devices (including headphones, laptops, and tablets) must be returned within 14 days. A 15% restocking fee applies to all opened electronics returns."
    },
    {
        "id": 4, 
        "text": "Final Sale: Items marked as Final Sale or Clearance are sold as-is and cannot be returned, exchanged, or refunded under any circumstances."
    }
]

# 2. Define a search tool matching query terms against policies. Handles tool execution. Search policy store action. 
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

# 3. Register the tool schema
tools = [
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

# 4. Agent Execution. Ask, run tool, return response, feedback to model.
MODEL = "claude-opus-5-5"
client = AsyncAnthropic()
SYSTEM_PROMPT = (
    "You are a customer support assistant. Respond to user questions by calling search_policies to find return policies before answering questions about returns or refunds."
)
async def run_rag_agent(question):
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

            if tool_name == "search_policies":
                try:
                    outcome = search_policies(query=tool_args["query"])
                    is_error = False
                except Exception as e:
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

        # 4. Record - feed the results back to the context, then return to step 1
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"Stopped: reached {max_turns} turns before Claude finished.")  
        
    return messages

query = "I bought headphones 10 days ago and opened them, but they don't fit. Can I return them for a refund?"
if __name__ == "__main__":
    asyncio.run(run_rag_agent(query)) 


