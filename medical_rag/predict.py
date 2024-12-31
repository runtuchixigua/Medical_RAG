from langchain import PromptTemplate
from get_vector import *
from model import ChatGLM_MD
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = r'D:\AIproject\Pycharmproject\LLM\Custom_llm\moka-ai\m3e-base'
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local('faiss', embeddings,allow_dangerous_deserialization=True)


def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content.replace('\n\n', '\n'))
    return "\n".join(related_content)


def define_prompt():
    question = '在浴室肾虚怎么办'
    docs = db.similarity_search(question, k=1)
    # print(docs)
    related_content = get_related_content(docs)
    # print(related_content)

    PROMPT_TEMPLATE = """
        基于以下已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分。
        已知内容:
        {context}
        问题:
        {question}"""
    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=PROMPT_TEMPLATE,
    )
    # print(prompt)

    my_pmt = prompt.format(context=related_content,
                        question=question)

    return my_pmt


def qa():
    llm = ChatGLM_MD()
    llm.load_model(r'D:\modelfile\chatglm3-6b')
    my_pmt = define_prompt()
    result = llm(my_pmt)
    return result

if __name__ == '__main__':
    result = qa()
    print(result)


