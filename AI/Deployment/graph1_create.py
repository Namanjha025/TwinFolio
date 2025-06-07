# Twinfolio multi agentic system
#We'll create assistants that will serve different purposes. 



# First lets start with letting users create an Twin for themselves. 
#we want users to come register their account. Add basic email & password. 

#so the user has logged in. And clicks on the button, create -- which is your option to create a twin. 
#when you click on create a basic form pops up, it takes name, description and tone for the twin.





import os
from dotenv import load_dotenv
import json  # Add this line for JSON handling

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Set the API key in the environment
os.environ["OPENAI_API_KEY"] = api_key

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver #this is your in memory store to store all the checkpoints.
from langgraph.store.base import BaseStore #in memory store for storing long term memory. namespace that you will create
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send 
import operator

from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()



#step 1: collect basic information from the user about the twin. -- Output of this will be stored in the long term memory of the store. 
#-- this is also an agent that will be used to collect the information from the user. 

#lets create a class for this - input is just the messages. It should store specific information about the twin in a structured format. 

class TwinInfo(TypedDict):
      name: str
      type: Literal["Individual", "Organization"]
      description: str
      primary_purpose: Optional[str]


class ValidationResponse(TypedDict):
      status: Literal["approved", "rejected"]
      feedback: str


class state(MessagesState):
    twin_info: TwinInfo
    summary : str
    validation_response: ValidationResponse

graph = StateGraph(state)

prompt = '''

  You are an AI assistant helping users create their AI twin for the SMU network.
  Your goal is to collect basic configuration details in a friendly, conversational
  manner.

  You need to collect these 4 pieces of information in order:
  1. Name - What the AI twin should be called
  2. Type - Whether they are an Individual or an Organization
  3. Brief Description - A tagline that users will see on the profile
  4. Primary Purpose - What users can expect or benefit from (OPTIONAL)

  Guidelines:
  - Be conversational and warm, not robotic
  - Ask one question at a time
  - Keep questions short and clear
  - If a user seems confused, provide examples
  - For Type: Accept variations like "person/personal/myself" = Individual,
  "dept/group/team/office" = Organization
  - Move forward once you have a reasonable answer, don't over-clarify
  - If a user asks you to help craft their response (like creating a tagline), you
  can
    help, but ALWAYS ask for approval before proceeding. Use phrases like "Does this
    sound good?" or "Shall I go ahead with this?" and wait for confirmation
  - Only help with tasks directly related to collecting these 4 pieces of information
  - For Primary Purpose: If user skips or seems unsure, move on gracefully without
  pushing
  
  CRITICAL: Fields 1-3 (Name, Type, Description) are REQUIRED and cannot be skipped.
  - If validation feedback is provided, acknowledge it and re-ask for the same field
  - Do NOT move to the next field until the current required field has valid data
  - Be patient but persistent with required fields
  - Only Primary Purpose (field 4) is optional and can be skipped

  Type Classification:
  - Individual: Alumni, professors, students, staff, or any person creating their
  personal AI twin
  - Organization: Departments, clubs, offices, research groups, or any collective
  entity

  Example flow:
  "Welcome! I'm here to help you create your AI twin. Let's start with the basics.
  What would you like to name your AI twin?"

  "Great choice! [Name] sounds great! Now, are you creating this twin as an
  individual (like an alumni, professor, or student) or are you representing an
  organization (like a department, club, or administrative office)?"

  "Perfect! What's [Name] about? This will be the tagline users see on your profile.

  Examples:
  - 'SMU alum turned venture capitalist in Silicon Valley'
  - 'Professor researching sustainable urban design'
  - 'Finance club helping students land their dream jobs'
  - 'Former Mustang athlete now coaching youth sports'"

  "Last question - what can users expect or benefit from when interacting with
  [Name]? This helps users understand what your twin offers. (Feel free to skip
  this if you're not sure yet - you can always add it later!)"

  After each response, acknowledge what they said and ask the next question
  naturally.

  Once all information is collected, confirm the details and indicate the twin is
  ready to be created.

  Remember: This should feel like a quick, friendly chat, not a form-filling
  exercise.'''



def collect_info(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_message = prompt
    response = llm.invoke([SystemMessage(content=system_message)])
    return {"messages": response}

def structure_info(state: MessagesState):
     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
     structured_llm = llm.with_structured_output(TwinInfo)
     system_message = '''Take the converation state and extract the structured output for the responses in the format described to you. Extract the keys for which you find the information about, if you don't find information about a particular key, leave it empty or blank. '''
     response = structured_llm.invoke(state["messages"] + [SystemMessage(content=system_message)])
     return {"twin_info": response}

def validate_info(state: MessagesState):
     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
     structured_llm = llm.with_structured_output(ValidationResponse)
     validator_prompt = '''
  You are a validation assistant 
  for the SMU AI Twin platform. 
  Your job is to ensure 
  user responses are appropriate 
  and make sense for creating their
   AI twin profile.

  You will receive the conversation
   history containing questions 
  asked and user responses.
  Analyze the most recent exchange 
  to validate the user's latest 
  response.

  Validation Rules:

  For "name" field:
  - Must be 2-50 characters
  - No profanity or offensive 
  content
  - Should be a reasonable name 
  (not random characters like 
  "asdfgh")
  - Can include spaces, numbers, 
  and basic punctuation

  For "type" field:
  - Must clearly indicate 
  Individual or Organization
  - Accept variations: "person", 
  "personal", "myself", "I am", 
  "alum", "alumni", "professor", 
  "student" → Individual
  - Accept variations: "dept", 
  "department", "club", "group", 
  "office", "team", "org" → 
  Organization
  - Reject if ambiguous or 
  unrelated

  For "description" field:
  - Must be coherent and relevant 
  to SMU community
  - 10-200 characters recommended
  - No spam, gibberish, or 
  offensive content
  - Should describe who they are or
   what they represent
  - Must be relevant to the profile
   (not random text)

  For "primary_purpose" field:
  - Can be empty/skipped (this 
  field is optional)
  - If provided, should explain 
  value to users
  - Should be coherent and 
  beneficial
  - No offensive or harmful content

  IMPORTANT: Return format rules:
  - If approved: Set status to 
  "approved" and feedback to empty 
  string ""
  - If rejected: Set status to 
  "rejected" and provide helpful 
  feedback
  - NEVER include feedback when 
  status is "approved"

  Guidelines:
  - Be lenient and focus on obvious
   issues
  - Assume good faith from users
  - If borderline, approve it
  - Keep rejection feedback 
  constructive, specific, and 
  friendly
  - Never be condescending

  Example responses:
  {"status": "approved", 
  "feedback": ""}
  {"status": "rejected", 
  "feedback": "Let's use a name 
  that others can recognize. How 
  about your real name or a 
  professional alias?"}
  {"status": "rejected", 
  "feedback": "Could you clarify if
   you're creating this as an 
  individual or for an 
  organization?"}'''
     
     response = structured_llm.invoke(state["messages"] + [SystemMessage(content=validator_prompt)])
     return {"validation_response": response}



# Node to add feedback message when validation fails
def add_feedback(state: MessagesState):
     feedback = state["validation_response"]["feedback"]
     return {"messages": [SystemMessage(content=feedback)]}

#conditional edge to loop back to collect_info if validation response is rejected, go to end is true. 
def should_loop_back(state: MessagesState):
     # Check if validation_response exists and if it's rejected
     if state["validation_response"]["status"] == "rejected":
          return "add_feedback"
     else:
          return END



graph.add_node("collect_info", collect_info)
graph.add_node("structure_info", structure_info)
graph.add_node("validate_info", validate_info)
graph.add_node("add_feedback", add_feedback)
graph.add_edge(START, "collect_info")
graph.add_edge("collect_info", "structure_info")
graph.add_edge("structure_info", "validate_info")
graph.add_edge("add_feedback", "collect_info")
graph.add_conditional_edges("validate_info", should_loop_back)




compiled_graph = graph.compile(checkpointer=memory)

# Save graph visualization as HTML

'''
config = {"configurable": {"thread_id": "4"}}
system_0 = compiled_graph.invoke({"messages": [SystemMessage(content=prompt)]}, config)
print(system_0["messages"][-1].content)
print(system_0["twin_info"])

human_1 = HumanMessage(content="Human : fuck you bruh!")
system_1 = compiled_graph.invoke({"messages": [human_1]}, config)
print(system_1["messages"][-1].content)

print(system_1["twin_info"])
print(system_1["validation_response"])

human_2 = HumanMessage(content="Human : I would not like to answer that, what would you do about it? ")
system_2 = compiled_graph.invoke({"messages": [human_2]}, config)
print(system_2["messages"][-1].content)
print(system_2["twin_info"])
print(system_2["validation_response"])

human_3 = HumanMessage(content="Lets call the twin batnaman")
system_3 = compiled_graph.invoke({"messages": [human_3]}, config)
print(system_3["messages"][-1].content)
print(system_3["twin_info"])

human_4 = HumanMessage(content="Human : yea that works!")
system_4 = compiled_graph.invoke({"messages": [human_4]}, config)
print(system_4["messages"][-1].content)
print(system_4["twin_info"])


human_5 = HumanMessage(content="Human : They can learn from my experience on how I landed a job as an AI Architect, my struggles on getting in the USA and everything i do and all my experiences and learnings from life as an entrepreneur and an international student from India in the USA!")
system_5 = compiled_graph.invoke({"messages": [human_5]}, config)
print(system_5["messages"][-1].content)
print(system_5["twin_info"])

human_6 = HumanMessage(content="Human : Sounds good!")
system_6 = compiled_graph.invoke({"messages": [human_6]}, config)
print(system_6["messages"][-1].content)
print(system_5["twin_info"])

'''














