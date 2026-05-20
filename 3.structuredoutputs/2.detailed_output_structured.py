# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()   


model = ChatOpenAI(model="gpt-4-turbo", temperature=0.9)

class Review(TypedDict):
    key_themes: Annotated[list[str], "must write down all the important concept discussed int he review in a list"]
    summary:Annotated[str, "must write a summary of the review"]
    sentiment:Annotated[str, "must write the overall sentiment of the review. it should be either positive, negative or neutral"]
    pros:Annotated[Optional[list[str]], "must write a list of pros if any, otherwise write None"]
    cons:Annotated[Optional[list[str]], "must write a list of cons if any, otherwise write None"]

structured_model=model.with_structured_output(Review)

# promt="""This hardware is great, but the software feels kind of bloated.
# So may boilerplate apps. and my phone keeps hangin when i play PUBG"""

prompt="""
I purchased the Pixel 7a about a year ago, attracted above all by the possibility of recording calls natively, without having to resort to external solutions or third-party applications. Bitter discovery: This feature is only available to users in the United States. A serious lack of transparency on Google's part, which greatly affected my overall experience.

The only function that I initially found useful was that of the call filter: when a phone call arrives, in addition to the classic "answer" button, a "filter" option was available which activated an automatic voice to ask the interlocutor the reason for the call, transcribing the answer on the screen in real time. Too bad that this function was also unstable: the "filter" button sometimes appeared, sometimes it mysteriously disappeared.

The camera? Average, nothing exceptional. ents itself as a "smart" alternative in the medium-high range.

Another obvious flaw: during a call, if another one comes in, it is not possible to end the first one to answer the second one. The only option is to put the first call on hold, or ask the person you are talking to to hang up, which is frankly absurd in 2025.

In addition, I have had calls drop out for no reason several times, a situation that has created embarrassment and annoyance on several occasions."

"""

result=structured_model.invoke(prompt)

print(result)