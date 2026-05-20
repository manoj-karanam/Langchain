from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model=ChatOpenAI(model="gpt-4-turbo")

prompt1 = PromptTemplate(template="generate a short and simple notes from the following topic {topic}",
                            input_variables=["topic"])

prompt2 = PromptTemplate(template="generate 5 short question answer from the follwoing {text}",
                         input_variables=["text"])

prompt3= PromptTemplate(template="merge the provided notes and quiz into a single document \n notes: {notes} \n quiz: {quiz}",
                        input_variables=["notes", "quiz"])

parser = StrOutputParser()

runnable_chain=RunnableParallel(
    {
    "notes": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
    }
)

final_chain=prompt3 | model | parser

chains = runnable_chain | final_chain

text = """
This is a review for the iPhone 15 Pro Max, which I have been using for a few weeks now. Overall, I am quite disappointed with this phone. The only function that I initially found useful was that of the call filter: when a phone call arrives, in addition to the classic "answer" button, a "filter" option was available which activated an automatic voice to ask the interlocutor the reason for the call, transcribing the answer on the screen in real time. Too bad that this function was also unstable: the "filter" button sometimes appeared, sometimes it mysteriously disappeared.

"""


result = chains.invoke(text)
print(result)
