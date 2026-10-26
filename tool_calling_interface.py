'''AI Agents Core Mechanics: The Tool Calling Interface

The component within the core loop focusses on handling tool schemas to pass to the model API. The interaction is a
coordinated round trip between the stateless model and the stateful execution environment. 

Execution Cycle
1) Request Generation - Evaluate user query. Check if external data needed. Output structured tool_calls if needed
2) Native Execution -  Hosting application parses the arguments, executes the native function, caputes the return value 
3) Context Resumption - Append tool results to the conversation history as message. Queries again. Model produces final answer

Model extracts parameters from query natural language inut, maps them to the structured json schema, requests the tool.
Application performs the math and feedbacks the result enabling the model to deliver deterministc answers. 
'''

import asyncio
from anthropic import AsyncAnthropic

# 1) Native implementation of the tool

def calculate_loan_payment(principal, rate, years):
    print(f"  [EXECUTE] Running calculate_loan_payment with P={principal}, R={rate}, Y={years}")
    monthly_rate = rate / 12.0
    months = years * 12
    if monthly_rate == 0:
        payment = principal / months
    else:
        payment = principal * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
    return f"Monthly Payment: ${payment:.2f}"

# 2) Schema definition matching the function parameters 

tools = [{
        "name": "calculate_loan_payment",
        "description": "Calculates the monthly repayment of a loan given the principal, rate and years",
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {"type": "integer", "description": "The intial total loan value."},
                "rate": {"type": "number", "description": "The annual interest rate (e.g. 0.055 for 5.5%)"},
                "years": {"type": "integer", "description": "Duration of the loan in years"}
                },
            "required": ["principal", "rate", "years"]
            }
        }]

# 3) Agent execution. Ask, run tool, ask again.

MODEL = "claude-opus-5-5"
client = AsyncAnthropic()
SYSTEM_PROMPT = (
    "You are a loan repayment calculator. "
    "Use the calculate_loan_payment tool to calculate the monthly repayments of loans."
)

async def run_loan_repayment_agent(query):
    print(f"--- Starting Agent Loop for input: '{query}'---\n")

    # 1) Read the initial user query and store it in the messages list for short term context memory. 
    messages = [
        {"role": "user", "content": query}
    ]    

    # Turn 1: Query the model with the tools registered
    resp = await client.messages.create(
                model = MODEL,
                system = SYSTEM_PROMPT,
                messages=messages, # Query the language model with the context
                tools=tools,
                max_tokens=15000
            )
    
    messages.append({"role": "assistant", "content": resp.content})

    # resp.content is a list of blocks - typical reply looks like:
    '''
    [
        TextBlock(type="text", text="I'll calculate that for you."),
        ToolUseBlock(type="tool_use", id="toolu_1", name="calculate_loan_payment",
        input={"principal": 15000, "rate": 0.055, "years": 4}),
    ]
    '''

    if resp.stop_reason == "tool_use": 
        # Run each tool claude requests (expect only one)
        tool_results = []
        for block in resp.content: # INSPECT EACH BLOCK 
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_args = block.input
            print(f"Model requested tool: {tool_name} with args: {tool_args}")

            if tool_name == "calculate_loan_payment":
                outcome = calculate_loan_payment(
                    principal=tool_args["principal"],
                    rate = tool_args["rate"],
                    years = tool_args["years"]
                )
                is_error = False
            else:
                outcome = f"Error. Unknown tool {tool_name}"
                is_error = True

            # Feed the outcome back to the context 
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": outcome,
                "is_error": is_error
            })
        # Append the outcome back to the context
        messages.append({"role": "user", "content": tool_results})

        # Turn 2: Generate the final response 
        resp2 = await client.messages.create(
             model = "claude-opus-5-5",
            system = SYSTEM_PROMPT,
            messages=messages, # Query the language model with the context
            tools=tools,
            max_tokens=15000
        )
        messages.append({"role": "tool", "content": resp2.content})

    # Print Claude's final words
    for block in messages[-1]["content"]:
        if block.type == "text":
            print(block.text)

    return messages

query = "Compute my monthly payment for a loan of $15000 at 5.5% annual interest over 4 years."
if __name__ == "__main__":
    asyncio.run(run_loan_repayment_agent(query))