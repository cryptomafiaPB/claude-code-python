import argparse
import os
import sys
import json

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

############ TOOL DEFINITIONS ############

def Read(file_path: str) -> str:
    """Read and return content of a file"""
    print("Inside Read function", file=sys.stderr)
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return content of a file",
            "parameters": {
                "type": "object",
            "properties": {
                "file_path": {
                "type": "string",
                "description": "The path to the file to read"
                }
            },
            "required": ["file_path"]
        }
    }
    }
]


def main():
    p = argparse.ArgumentParser() # init
    p.add_argument("-p", required=True) # add argument flag
    args = p.parse_args() # generate args object
    
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        max_completion_tokens=5000,
        tools=tools
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: Uncomment the following line to pass the first 
    if chat.choices[0].message.tool_calls:
        if chat.choices[0].message.tool_calls[0].function.name == "Read":
            file_path = json.loads(chat.choices[0].message.tool_calls[0].function.arguments)
            content = Read(file_path["file_path"])
            print(content)
    else:
        print(chat.choices[0].message.content)

    ############### Local testing code ################
    # i = input("Enter your message: ")
    # print(f"You entered: {i}")


if __name__ == "__main__":
    main()
