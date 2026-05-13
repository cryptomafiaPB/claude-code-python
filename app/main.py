import argparse
from json import tool
import os
import subprocess
import sys
import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageParam

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

max_iteration = 12
current_iteration = 0

############ TOOL DEFINITIONS ############

def read_file(file_path: str) -> str:
    """Read and return content of a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            f.close()
            return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """Write content to a file"""
    try:
        # If file does not exist, create it & if it does exist, overwrite it with new content
        with open(file_path, 'w') as f:
            f.write(content)
            f.close()
        
        return "File written successfully"
    except Exception as e:
        return f"Error writing file: {str(e)}"
    
def run_bash_cmd(command: str) -> str:
    """Execute the shell command"""
    try:
        # parse command
        cmd = command.split()
        print(command, cmd)
        # Run subprocess
        result = subprocess.run(
            cmd,  # list of command and args
            capture_output=True,  # Saves stdout and stderr internally
            text=True,  # Returns strings instead of bytes
            check=True  # Raises CalledProcessError if command fails
        )
        print(result.stdout.strip())
        return result.stdout.strip()
    except Exception as e:
        print(e)
        return f"Error executing command {command}: {str(e)}"
    

################ HELPER FUNCTIONS ################

def call_LLM(messages: list[ChatCompletionMessageParam]) -> Any:
    """Call the LLM with the given messages and return the response"""

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            max_completion_tokens=5000,
            tools=tools
        )
    
    if not response.choices or len(response.choices) == 0:
        raise RuntimeError("no choices in response")

    return response

def get_tool_response(tool_call) -> dict[str, Any]:
    """Execute the tool call and return the response"""
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)


    tool_response = TOOL_MAP[tool_name](**tool_args)

    # print(f"Tool call: {tool_name} with args {tool_args} and tool call ID {tool_call.id}", file=sys.stderr)
    return {
        "tool_name": tool_name,
        "tool_call_id": tool_call.id,
        "content": json.dumps(tool_response)
    }


################ UTILS ################

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
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
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash_cmd",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

TOOL_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "run_bash_cmd": run_bash_cmd
}

messages: list[ChatCompletionMessageParam] = []



def main():
    global current_iteration

    p = argparse.ArgumentParser() # init
    p.add_argument(
        "-p",
        required=True,
        help="Prompt to send to the Claude model"
    ) # add argument flag
    args = p.parse_args() # generate args object

    # Append user message to messages list
    messages.append({"role": "user", "content": args.p})

    # Agentic Loop
    while current_iteration < max_iteration:

        current_iteration += 1
        chat = call_LLM(messages)

        if chat.choices[0].message.tool_calls is not None:
            # Append assistant's response with tool calls
            messages.append({
                "role": "assistant",
                "content": chat.choices[0].message.content,
                "tool_calls": chat.choices[0].message.tool_calls
            })
            
            # Append tool result for each tool call
            tool_calls = chat.choices[0].message.tool_calls
            for tool_call in tool_calls:
                tool_response = get_tool_response(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_response["tool_call_id"],
                    "content": tool_response["content"]
                })
        else:
            messages.append({
                "role": "assistant",
                "content": chat.choices[0].message.content
            })

            print(chat.choices[0].message.content)
            break


if __name__ == "__main__":
    main()
