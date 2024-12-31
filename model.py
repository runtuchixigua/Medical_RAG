from typing import Optional, List  # 导入类型提示，用于注解方法和函数参数类型
from langchain_core.language_models.llms import LLM  # 从Langchain库导入基类LLM
from langchain_community.llms.utils import enforce_stop_tokens  # 导入用于确保停止令牌的实用函数
from transformers import AutoTokenizer, AutoModel  # 从Transformers库导入用于加载模型和分词器的类


class ChatGLM_MD(LLM):  # 定义ChatGLM_MD类，继承自LLM类
    # 设置默认的模型参数
    max_token: int = 4096  # 最大token数（生成回答时的最大长度）
    temperature: float = 0.8  # 生成的文本的“随机性”，越高越随机
    top_p: float = 0.9  # Top-p采样，用于控制生成文本的多样性
    tokenizer: object = None  # 用于分词的tokenizer
    model: object = None  # 模型本身
    history: List = []  # 用于存储对话历史，以便上下文关联

    def __init__(self):
        super().__init__()  # 初始化父类

    @property
    def _llm_type(self) -> str:
        # 该属性返回该模型类型的字符串，通常是模型的名字
        return 'ChatGLM_MD'

    def load_model(self, model_path):
        """
        加载预训练的模型和分词器。
        参数:
            model_path (str): 模型文件夹的路径，其中包含预训练的模型和分词器。
        """
        # 加载模型并将其移到GPU上，转换为半精度以提高性能
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        # 加载对应的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        生成模型的回答。
        参数:
            prompt (str): 输入给模型的提示（问题或对话内容）
            stop (Optional[List[str]]): 可选的停止令牌列表，当输出包含这些令牌时停止生成
        返回:
            str: 模型生成的响应
        """
        # 调用模型进行对话生成，历史记录会被传入，以保持上下文
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p
        )

        # 如果提供了停止令牌，则应用停止令牌
        if stop is not None:
            response = enforce_stop_tokens(response, stop)

        # 更新历史记录
        self.history = self.history + [[history, response]]

        # 返回生成的响应
        return response


if __name__ == '__main__':
    # 创建ChatGLM_MD模型的实例
    model = ChatGLM_MD()
    # 加载预训练的ChatGLM模型，模型文件路径根据实际情况设置
    model.load_model('D:\modelfile\chatglm3-6b')

    # 使用加载的模型进行对话，输入“您好！”
    model('您好！')
