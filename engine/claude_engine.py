import os
import anthropic


MODEL_ALIAS = {
    "claude-code": "claude-sonnet-4-5-20250929",
}

class ClaudeEngine:
    def __init__(self, llm_engine_name: str):
        if llm_engine_name in MODEL_ALIAS:
            llm_engine_name = MODEL_ALIAS[llm_engine_name] # 如果是别名，替换为正式名称
        self.model = llm_engine_name
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ANTHROPIC_API_BASE_URL")
        if not api_key:
            raise ValueError("没有检测到 ANTHROPIC_API_KEY 环境变量，请先设置它。")
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def respond(self, user_input: str, temperature: float = 0.1) -> str:
        """
        使用 Anthropic Messages API 生成 Claude 的回复。
        """
        try:
            print(f"正在调用 Claude API，使用的模型: {self.model}")
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                temperature=temperature, 
                messages=[
                    {"role": "user", "content": user_input[0]["content"]}
                ]
            )
            print(f"Claude API 调用成功，使用的模型: {self.model}")
            return response.content[0].text
        except Exception as e:
            print(f"Claude API 调用出错: {e}")
            return ""  # 出错时返回空字符串
