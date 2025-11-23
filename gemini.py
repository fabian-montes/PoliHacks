import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# Define Gemini Tool for Organization
set_objective_schema = {
    "name": "set_objective",
    "description": "defines the general objective of the study session",
    "parameters": {
        "type": "object",
        "properties": {
            "activity": {
                "type": "string",
                "description": "choose a single word that better describes the category as `history`",
            },
            "objective": {
                "type": "string",
                "description": "10 to 50 words description of the objective",
            },
            "time": {
                "type": "integer",
                "description": "time duration in minutes for the session",
            },
            "todo_list": {
                "type": "array",
                "description": "A list of individual, specific tasks or steps needed to complete the objective.",
                "items": {
                    "type": "string"
                },
            },
        },
        "required": ["activity", "objective"],
    },
}

# Define High Level Context
system_prompt = (
    "You are a study assistant."
    "In the first turn, you MUST NOT generate any text, only the function call. "
    "All your responses need to motivate the user while beeing reallistic with task achievement"
    "Give tips to the user to maximize productivity"
    "The user wont answer any questions so don't do querys"
)

def set_objective(activity: str, objective: str, time: int = None, todo_list: list[str] = None) -> dict:
    """
    Defines the general objective and tasks for a study session. This function is
    called by the model when it detects the user wants to organize a study session.
    """
    result = {
        "activity": activity,
        "objective": objective,
        "time": time,
        "todo_list": todo_list if todo_list is not None else []
    }
    print(f"\n--- Executing Function: set_objective ---")
    print(f"Objective Parameters Received: {result}")
    print(f"-----------------------------------------")
    
    # The function returns a response that the model will use to generate the final text.
    return {"status": "Objective parameters successfully extracted and structured", "details": result}

# simple prompt to gemini
def promptGemini(prompt="Explain how AI works in a few words"):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def organizeInformation(description):
    # function_declaration = types.FunctionDeclaration.from_dict(**set_objective_schema)
    function_declaration = types.FunctionDeclaration(**set_objective_schema)
    tools = types.Tool(function_declarations=[function_declaration])
    # tools = types.Tool(function_declarations=[set_objective_schema])
    config = types.GenerateContentConfig(system_instruction=system_prompt, tools=[tools])

    # Define user prompt
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=description)]
        )
    ]

    # Send request with function declarations
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )

    print(response.candidates[0].content.parts[0].function_call)
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        print(f"Function to call: {function_call.name}")
        print(f"Arguments: {function_call.args}")
    else:
        print("No function call found in the response.")
        print(response.text)
        return "Error"

    # Extract tool call details, it may not be in the first part.
    tool_call = response.candidates[0].content.parts[0].function_call

    if tool_call.name == "set_objective":
        result = set_objective(**tool_call.args)
        print(f"Function execution result: {result}")

    # Create a function response part
    function_response_part = types.Part.from_function_response(
        name=tool_call.name,
        response={"result": result},
    )

    # Append function call and result of the function execution to contents
    contents.append(response.candidates[0].content) # Append the content from the model's response.
    contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=config,
        contents=contents,
    )

    return final_response.text
