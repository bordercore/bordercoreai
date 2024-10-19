import importlib
import json
import random
import re
import string

from modules.exceptions import JsonParsingError, LLMResponseError


class FunctionCall():

    def __init__(self, model_name, **args):
        self.model_name = model_name
        self.args = args

    def generate_random_id(self, length=6):
        characters = string.ascii_letters + string.digits
        random_id = "".join(random.choice(characters) for _ in range(length))
        return random_id

    def rename_key(self, data, old_key, new_key):
        if old_key in data:
            data[new_key] = data.pop(old_key)

    def call_function_from_json(self, messages, response):
        json_match = re.search(r"<\|python_tag\|>(.*?)<\|eom_id\|>", response)

        if not json_match:
            raise LLMResponseError(f"No JSON found in the LLM response: {response}")

        tool_call = None
        json_string = json_match.group(1)
        try:
            tool_call = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise JsonParsingError(f"Error decoding JSON: {e}")

        tool_call_id = self.generate_random_id()

        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": tool_call
                    }
                ]
            }
        )

        # The chat template expects "arguments", but Llama-3.1 produces "parameters"
        #  instead. We need to change this to avoid this Jinja error:
        #  TypeError: Object of type Undefined is not JSON serializable
        self.rename_key(messages[-1]["tool_calls"][0]["function"], "parameters", "arguments")

        func_name = tool_call["name"]
        parameters = tool_call["arguments"]

        main_module = importlib.import_module(f"modules.{self.tool_name}")

        if not hasattr(main_module, func_name):
            raise AttributeError(f"Function '{func_name}' not found in module '{self.tool_name}'")

        func = getattr(main_module, func_name)

        if not callable(func):
            raise TypeError(f"'{func_name}' is not callable")

        result = func(**parameters)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": func_name,
                "content": str(result)
            }
        )

        from modules.chatbot import ChatBot
        chatbot = ChatBot(self.model_name, temperature=0.1)
        response = chatbot.send_message_to_model(messages, replace_context=True, tool_name=self.tool_name, tool_list=self.tool_list)
        content = ChatBot.get_streaming_message(response)

        # Remove trailing <|eot_id|> token.
        eot_id = "<|eot_id|>"
        if content.endswith(eot_id):
            content = content[:-len(eot_id)]

        return content

    def choose_function(self, messages):
        from modules.chatbot import ChatBot
        chatbot = ChatBot(self.model_name)
        response = chatbot.send_message_to_model(messages, replace_context=True, tool_name=self.tool_name, tool_list=self.tool_list)
        return ChatBot.get_streaming_message(response)

    def run(self, prompt):
        prompt += f"{prompt} Please don't tell me how you got the answer, I only want the answer."
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can use functions when necessary. When you receive a tool call response, use the output to format an answer to the orginal user question."},
            {"role": "user", "content": prompt}
        ]

        target_function = self.choose_function(messages)
        return "Using Wolfram Alpha. " + self.call_function_from_json(messages, target_function)
