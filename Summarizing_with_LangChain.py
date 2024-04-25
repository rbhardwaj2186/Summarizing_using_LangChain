import os

import tiktoken
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### A) Basic Prompt
from langchain_openai import ChatOpenAI
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
text = """ Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, particularly computer systems. It is a field of research in computer science that develops and studies methods and software which enable machines to perceive their environment and uses learning and intelligence to take actions that maximize their chances of achieving defined goals.[1] Such machines may be called AIs.

AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., ChatGPT and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go).[2] However, many AI applications are not perceived as AI: "A lot of cutting edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore."[3][4]

Alan Turing was the first person to conduct substantial research in the field that he called machine intelligence.[5] Artificial intelligence was founded as an academic discipline in 1956.[6] The field went through multiple cycles of optimism,[7][8] followed by periods of disappointment and loss of funding, known as AI winter.[9][10] Funding and interest vastly increased after 2012 when deep learning surpassed all previous AI techniques,[11] and after 2017 with the transformer architecture.[12] This led to the AI boom of the early 2020s, with companies, universities, and laboratories overwhelmingly based in the United States pioneering significant advances in artificial intelligence.[13]

The growing use of artificial intelligence in the 21st century is influencing a societal and economic shift towards increased automation, data-driven decision-making, and the integration of AI systems into various economic sectors and areas of life, impacting job markets, healthcare, government, industry, and education. This raises questions about the long-term effects, ethical implications, and risks of AI, prompting discussions about regulatory policies to ensure the safety and benefits of the technology.

The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and support for robotics.[a] General intelligence—the ability to complete any task performable by a human on an at least equal level—is among the field's long-term goals.[14]

To reach these goals, AI researchers have adapted and integrated a wide range of techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[15]

Goals
The general problem of simulating (or creating) intelligence has been broken into sub-problems. These consist of particular traits or capabilities that researchers expect an intelligent system to display. The traits described below have received the most attention and cover the scope of AI research.[a]

Reasoning and problem solving
Early researchers developed algorithms that imitated step-by-step reasoning that humans use when they solve puzzles or make logical deductions.[16] By the late 1980s and 1990s, methods were developed for dealing with uncertain or incomplete information, employing concepts from probability and economics.[17]

Many of these algorithms are insufficient for solving large reasoning problems because they experience a "combinatorial explosion": they became exponentially slower as the problems grew larger.[18] Even humans rarely use the step-by-step deduction that early AI research could model. They solve most of their problems using fast, intuitive judgments.[19] Accurate and efficient reasoning is an unsolved problem.

Knowledge representation

An ontology represents knowledge as a set of concepts within a domain and the relationships between those concepts.
Knowledge representation and knowledge engineering[20] allow AI programs to answer questions intelligently and make deductions about real-world facts. Formal knowledge representations are used in content-based indexing and retrieval,[21] scene interpretation,[22] clinical decision support,[23] knowledge discovery (mining "interesting" and actionable inferences from large databases),[24] and other areas.[25]  """

messages = [
    SystemMessage(content='You are an expert copywriter with expertize in summarizing documents'),
    HumanMessage(content=f'Please provide a short concise summary of the following text:\n TEXT: {text}')
]

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

print(f'The number of tokens in the prompt is: {llm.get_num_tokens(text)})')

system_output = llm(messages)
print(system_output.content)

# Adding a dynamic part to the prompt and will summarize using Prompt Templates
from langchain import PromptTemplate
from langchain.chains import LLMChain

template = '''
Write a concise and short summary of the following text:
TEXT: `{text}`
Translate the summary to {language}
'''
prompt = PromptTemplate(
    input_variables = ['text', 'language'],
    template=template
)

print(llm.get_num_tokens(prompt.format(text=text, language='English')))

chain = LLMChain(llm=llm, prompt=prompt)
summary = chain.invoke({'text': text, 'language': 'hindi'})
print(summary)

# Summarizing using StuffDocumentChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

with open('./sj.txt', encoding='utf-8') as f:
    text = f.read()
# text

docs = [Document(page_content=text)]
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

template = '''Write a concise and short summary of the following text 
TEXT:`{text}`
'''

prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

chain = load_summarize_chain(
    llm,
    chain_type='stuff',
    prompt=prompt,
    verbose=False
)
output_summary = chain.invoke(docs)

print(output_summary)

## Summarizng Large Documents Using map_reduce
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('./sj.txt', encoding='utf-8') as f:
    text = f.read()

# text

docs = [Document(page_content=text)]
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

llm.get_num_tokens(text)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.create_documents([text])

print(f'length of chunks is: {len(chunks)}')

chain = load_summarize_chain(
    llm,
    chain_type='map_reduce',
    verbose=False
)
output_summary = chain.invoke(chunks)
print(output_summary)

print(chain.llm_chain.prompt.template)
print(chain.combine_document_chain.llm_chain.prompt.template)

# map_reduce with Custom Prompts
map_prompt = '''
Write a short concise summary of the following:
Text: `{text}`
CONCISE SUMMARY:
'''
map_prompt_template = PromptTemplate(
    input_variables = ['text'],
    template=map_prompt
)

combine_prompt = '''
Write a concise summary of the following text that covers the key points.
Add a title to the summary.
Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE
Text: `{text}`
'''
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text'])
summary_chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    map_prompt=map_prompt_template,
    combine_prompt=combine_prompt_template,
    verbose=False

)
output = summary_chain.invoke(chunks)
print(output)

# Summarization using refine Chain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

loader = UnstructuredPDFLoader('./attention_is_all_you_need.pdf')
data = loader.load()
#print(data[0].page_content)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.002:.6f}')

print_embedding_cost(chunks)

chain = load_summarize_chain(
    llm=llm,
    chain_type='refine',
    verbose=False
)
output_summary = chain.invoke(chunks)

print(output_summary)

## Refine with Custom Prompts
prompt_template = """Write a concise summary of the following extracting the key Information:
Text: `{text}`
CONCISE SUMMARY:"""
initial_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

refine_template = '''
Your job is to produce a final summary.
I have provided an existing summary up to a certain point: {existing_answer}.
Please refine the existing summary with some more context below.
--------
{text}
--------
Start the final summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.

'''
refine_template = PromptTemplate(template=refine_template, input_variables=['existing_answer', 'text'])

chain = load_summarize_chain(
    llm=llm,
    chain_type='refine',
    question_prompt=initial_prompt,
    refine_template=refine_template,
    return_intermediate_step=False
)
output_summary = chain.invoke(chunks)

print(output_summary)




