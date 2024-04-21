import os
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
<<<<<<< HEAD
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
=======
>>>>>>> 70387100eedcbcf30a34d8c74d20c87d0c39f989
