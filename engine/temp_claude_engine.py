import os

MODEL_ALIAS = {
    "temp-claude-code": "claude-sonnet-4-5-20250929",
}

import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

from openai import OpenAI

api_key = 'sk-hCreREYu5iEdIcYNfp44zUDQWELDmS8MaraKQj1vBpVBkM0W' 
base_url = "https://ca.bt6.top/v1"


class TClaudeEngine:
    def __init__(self, llm_engine_name: str):
        self.model = llm_engine_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def respond(self, user_input: str, temperature: float = 0.2) -> str:
        """
        使用 Anthropic Messages API 生成 Claude 的回复。
        """
        try:
            print(f"正在调用 Claude API，使用的模型: {self.model}")
            print("user_input:", user_input)
            response = self.client.chat.completions.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are an expert Python programming assistant that helps scientist users to write high-quality code to solve their tasks.\nGiven a user request, you are expected to write a complete program that accomplishes the requested task and save any outputs in the correct format.\nPlease wrap your program in a code block that specifies the script type, python. For example:\n```python\nprint('Hello World!')\n```\n\nPlease keep your response concise and do not use a code block if it\'s not intended to be executed.\nPlease do not suggest a few line changes, incomplete program outline, or partial code that requires the user to modify.\nPlease do not use any interactive Python commands in your program, such as `!pip install numpy`, which will cause execution errors."},
                    {"role": "user", "content": user_input}
                ]
            )
            print(f"TClaude API 调用成功，使用的模型: {self.model}")
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Claude API 调用出错: {e}")
            return ""  # 出错时返回空字符串
