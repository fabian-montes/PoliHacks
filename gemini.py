import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

# Define Gemini Tool for Organization
set_objective_prompt = {
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
    "All your responses need to motivate the user while beeing reallistic with task achievement"
    "Give tips to the user to maximize productivity"
    "The user wont answer any questions so don't do querys"
)

# def set_objective(activity: str, objective: str) -> dict:
#     return {"activity": activity, "objective": objective}

# simple prompt to gemini
def promptGemini(prompt="Explain how AI works in a few words"):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def organizeInformation(description):
    tools = types.Tool(function_declarations=[set_objective_prompt])
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
        contents=description,
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

    if tool_call.name == "set_objective_prompt":
        result = set_light_values(**tool_call.args)
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
