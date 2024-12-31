from typing import Optional, List

from langchain_core.language_models.llms import LLM
from langchain_community.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel


class ChatGLM_MD(LLM):
    max_token: int = 4096
    temperature: float = 0.8
    top_p: float = 0.9
    tokenizer: object = None
    model: object = None
    history: List = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return 'ChatGLM_MD'

    def load_model(self, model_path):
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p
        )
        # print(response)
        # print(_)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[history, response]]
        return response


if __name__ == '__main__':
    model = ChatGLM_MD()
    model.load_model('D:\modelfile\chatglm-6b')
    model('您好！')
