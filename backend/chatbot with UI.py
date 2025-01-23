import chainlit as cl
from PIL import Image
import io
import os 
import google.auth
from dotenv import load_dotenv 
from typing import Optional
from dataclasses import dataclass
from  langchain_google_community import VertexAISearchRetriever,VertexAIMultiTurnSearchRetriever
from google.cloud.discoveryengine import SearchServiceClient 
from langchain.tools import Tool 
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq 
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union 
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence, Optional, Dict, Any
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
import pprint
from rich import print
import time


load_dotenv() 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

credentials, project_id = google.auth.default() 


import yaml

def load_config(file_path: str):
    try:
        with open(file_path, "r") as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error in YAML file: {exc}")
        return None

config = load_config("config.yaml")
GROQ_API_KEY = config['api_keys']['GROQ_API_KEY'] 
OPENAI_API_KEY = config['api_keys']['OPENAI_API_KEY']
LANGCHAIN_PROJECT = config['LANGCHAIN_PROJECT']
LANGCHAIN_API_KEY = config['LANGCHAIN_API_KEY']


retriever = VertexAIMultiTurnSearchRetriever(project_id=PROJECT_ID, location_id=LOCATION_ID, data_store_id=DATA_STORE_ID,get_extractive_answers=False) 
retriever2 = VertexAIMultiTurnSearchRetriever(project_id = PROJECT_ID,location_id = LOCATION_ID, data_store_id = DATA_STORE_ID2,get_extractive_answers=False)
retriever3 = VertexAIMultiTurnSearchRetriever(project_id = PROJECT_ID,location_id = LOCATION_ID4, data_store_id = DATA_STORE_ID4,get_extractive_answers=False)

retriever_tool_1 = create_retriever_tool(
    retriever,
    "retrieve_amway_blog_posts",
    "Search and return information about Amway US blog posts on Face serum guide, gut reset and gut health.",
)
retriever_tool_2 = create_retriever_tool(
    retriever2,
    "retrieve_WHO_posts",
    "Search and return information from the WHO articles. ",
)

retriever_tool_3 = create_retriever_tool(
    retriever3,
    "amway_sales_plan",
    "Search and return information from the amway core sales plan documnents",
)

tools = [retriever_tool_1,retriever_tool_3] 
model = SentenceTransformer('all-MiniLM-L6-v2') 

def retrieve_from_tool(tool, query: str) -> List[Union[Document, str]]:
    return tool.invoke({"query": query})

def compute_similarity(query_embedding, doc_embedding):
    return np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))

def get_content(doc: Union[Document, str]) -> str:
    if isinstance(doc, Document):
        return doc.page_content
    return doc  # If it's already a string, return it as is

def rank_documents(docs: List[Union[Document, str]], query: str) -> List[Union[Document, str]]:
    query_embedding = model.encode([query])[0]
    
    doc_similarities = []
    for doc in docs:
        content = get_content(doc)
        doc_embedding = model.encode([content])[0]
        similarity = compute_similarity(query_embedding, doc_embedding)
        doc_similarities.append((similarity, doc))
    
    # Sort documents by similarity score in descending order
    ranked_docs = [doc for _, doc in sorted(doc_similarities, key=lambda x: x[0], reverse=True)]
    return ranked_docs


class ImageInput(BaseModel):
    """Represents an uploaded image."""
    filename: str

class UserInput(BaseModel):
    """Represents user input, which can include text and/or an image."""
    text: str
    image: Optional[ImageInput] = None

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_docs: str
    user_input: UserInput

import requests

def get_token(auth_url, client_id, client_secret):
    auth_data = {
        "grant_type": "client_credentials",
        "client_secret": client_secret,
        "client_id": client_id
    }
    response = requests.post(auth_url, json=auth_data)
    response.raise_for_status()
    token = response.json().get("access_token")
    return token

def process_input(state):
    """
    Process the initial user input, which may include both text and an image.
    """
    print("---PROCESS INPUT---")
    user_input = state["user_input"]
    
    # Create a message that includes information about the image if present
    content = user_input.text
    if user_input.image:
        content += f"\n[Attached image: {user_input.image.filename}]"
    
    return {
        "messages": [HumanMessage(content=content)],
        "user_input": user_input
    }

def retrieve(state: Dict) -> Dict:
    print("---RETRIEVE DOCUMENTS---")
    messages = state["messages"]
    question = messages[0].content
    with ThreadPoolExecutor(max_workers=len(tools)) as executor:
        future_to_tool = {executor.submit(retrieve_from_tool, tool, question): tool for tool in tools}
        #print(future_to_tool)
        all_docs = []
        for future in as_completed(future_to_tool):
            tool = future_to_tool[future]
            try:
                docs = future.result()
                #print("docs",docs)
                all_docs.append(docs)
            except Exception as exc:
                print(f'{tool} generated an exception: {exc}')

    # Rank and filter the documents
    #print("all docs",all_docs)
    ranked_docs = rank_documents(all_docs, question)
    
    # Return top N documents (adjust N as needed)
    top_n = 5
    #print("The ranked docs",ranked_docs[:top_n])
    return {"retrieved_docs": ranked_docs[:top_n]}


def grade_documents(state) -> Literal["generate", "analyze_meal_question"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")
    

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = ChatGroq(model = "gemma2-9b-it", temperature = 0,streaming=True)
    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = state["messages"][0].content
    docs = state["retrieved_docs"]
    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return {"messages": [HumanMessage(content="Documents are relevant.")]}
        return { "generate": "generate"}

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return {"messages": [HumanMessage(content="Documents are not relevant, So now we check for if it releated to mela-plate or not?")]}
        return {"incorrect_question":"incorrect_question"}
    
def generate(state):
    """
    Generate answer based on retrieved documents.
    """
    print("---GENERATE ANSWER---")
    question = state["messages"][0].content
    docs = state["retrieved_docs"]
    sources = []
    
    # If docs is a list of Document objects
    for i, doc in enumerate(docs):
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            source = doc.metadata.get('source', f'Source {i+1}')
        else:
            source = f'Source {i+1}'
        if source not in sources:
            sources.append(source)
    
    
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    model = ChatGroq(model = "gemma2-9b-it", temperature = 0)
    prompt = hub.pull("rlm/rag-prompt")
    chain = prompt | model | StrOutputParser()

    response = chain.invoke({"context": docs, "question": question})
    return {"messages": [HumanMessage(content=response)]}

def incorrect_question(state):
    """
    Handle case when the question is deemed incorrect or irrelevant.
    """
    print("---INCORRECT QUESTION---")
    return {"messages": [HumanMessage(content="The question appears to be incorrect or irrelevant to the available information.")]}


def analyze_meal_question(state) -> Literal["meal_image_scan", "meal_text_scan", "barcode", "incorrect_question"]:
    """
    Analyzes the user's question to determine if it's related to meal plate analysis
    and what type of API call is needed.

    Args:
        state (dict): The current state

    Returns:
        dict: An updated state with the decision
    """
    print("---ANALYZE MEAL QUESTION---")

    class QuestionAnalysis(BaseModel):
        """Analysis result for the user's question."""
        is_meal_related: bool = Field(description="Whether the question is related to meal analysis")
        api_call_type: str = Field(description="Type of API call needed: 'image_scan', 'text_query', 'barcode_scan', or 'none'")

    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = ChatGroq(model = "gemma2-9b-it", temperature = 0,streaming = True)
    llm_with_tool = model.with_structured_output(QuestionAnalysis)

    prompt = PromptTemplate(
        template="""Analyze the following user question related to meal plate analysis:

        User Question: {question}

        Determine if the question is related to meal plate analysis and what type of API call is needed.
        The question is meal-related if it asks about food items, nutritional information, or involves analyzing a meal image or barcode.

        Possible API call types:
        1. 'image_scan': If the question involves analyzing an image of food.
        2. 'text_query': If the question is about specific food items or nutritional information.
        3. 'barcode_scan': If the question involves scanning a barcode.
        4. 'none': If the question is not related to meal analysis.

        Provide your analysis as a JSON object with two fields:
        1. 'is_meal_related': A boolean indicating if the question is related to meal analysis.
        2. 'api_call_type': A string indicating the type of API call needed ('image_scan', 'text_query', 'barcode_scan', or 'none').
        """,
        input_variables=["question"],
    )

    chain = prompt | llm_with_tool

    question = state["messages"][0].content
    user_input = state["user_input"]
    analysis_result = chain.invoke({"question": question})

    if analysis_result.is_meal_related:
        if analysis_result.api_call_type == "image_scan" and user_input.image:
            return {"messages": [HumanMessage(content="image scan required.")]}
        elif analysis_result.api_call_type == "text_query":
            return {"messages": [HumanMessage(content="text query required.")]}
        elif analysis_result.api_call_type == "barcode_scan" and user_input.image:
            return {"messages": [HumanMessage(content="Barcode scan required.")]}
    
    return {"messages": [HumanMessage(content="Question is not related to meal analysis.")]}

def evaluate_api_response(api_response: Dict[str, Any], user_question: str) -> str:
    """
    Evaluate the API response based on the user's question.
    """
    
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
    model = ChatGroq(model = "gemma2-9b-it", temperature = 0)
    prompt = PromptTemplate(
        template="""
        You are provided with a proper response from the llm which is in form of a dictonary in {meal_analysis} 
        Now your role as a very professional analyst is to analyse this response and provide imformation which the user is seeking in his/her question.
        Please don't try to analyze the image only focus on the api_response which is being passed to you. 

        Meal Analysis: {meal_analysis}

        User Question: {user_question}

        Please provide a concise answer addressing the user's question based on the meal analysis results.
        """,
        input_variables=["meal_analysis", "user_question"]
    )
    chain = prompt | model | StrOutputParser() 
    result = chain.invoke({"meal_analysis": str(api_response), "user_question": user_question})
    return result


def meal_image_scan(state):
    """
    Handle case when the question is deemed incorrect or irrelevant.
    """
    print("---Scanning the image---")
    user_input = state["user_input"]
    user_question = state["messages"][0].content
    if user_input.image:
        TOKEN = get_token("https://api-dv.amwayglobal.com/rest/oauth2/v1/token","3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M","9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu")
        headers_o = {
                    "Authorization": f"Bearer {TOKEN}",
                    "x-hw-program": "mg_testing",
                    "x-abold": "mg_abo",
                    "x-mealtime": "",
                    "x-genai-vendor": "openai",
                    "x-country-code": "mg_testing"
                }
        with open(user_input.image.filename, 'rb') as img_file:
                files = {'meal_image': img_file}
                response = requests.post("https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/meal-scan", headers=headers_o, files=files, data={})
                
                #print("the response:",response.json())
                answer = evaluate_api_response(response.json(), user_question)
                return {"messages": [HumanMessage(content=answer)]}
    else:
        return {"messages": [HumanMessage(content="No image provided for analysis.")]}

def meal_text_scan(state):
    """
    Handle case when the question is deemed incorrect or irrelevant.
    """
    print("---Text scan --")
    user_input = state["user_input"] 
    user_question = state["messages"][0].content 
    data = {'meal_description': user_input.text}
    TOKEN = get_token("https://api-dv.amwayglobal.com/rest/oauth2/v1/token","3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M","9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu")
    headers_o = {
                    "Authorization": f"Bearer {TOKEN}",
                    "x-hw-program": "mg_testing",
                    "x-abold": "mg_abo",
                    "x-mealtime": "",
                    "x-genai-vendor": "openai",
                    "x-country-code": "mg_testing"
                }
    response = requests.post("https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/meal-scan", headers=headers_o, files={}, data=data)
    answer = evaluate_api_response(response.json(), user_question)
    return {"messages": [HumanMessage(content=answer)]}

def barcode(state):
    """
    Handle case when the question is deemed incorrect or irrelevant.
    """
    print("---Barcode---")
    user_input = state["user_input"]
    user_question = state["messages"][0].content
    if user_input.image:
        TOKEN = get_token("https://api-dv.amwayglobal.com/rest/oauth2/v1/token","3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M","9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu")
        headers_o = {
                    "Authorization": f"Bearer {TOKEN}",
                    "x-hw-program": "mg_testing",
                    "x-abold": "mg_abo",
                    "x-mealtime": "",
                    "x-genai-vendor": "openai",
        }
        with open(user_input.image.filename, 'rb') as img_file:
            upc_file = {'image': img_file}
            response = requests.post("https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/upc", files=upc_file, headers=headers_o, data={})
            answer = evaluate_api_response(response.json(), user_question)
            return {"messages": [HumanMessage(content=answer)]}
    else:
        return {"messages": [HumanMessage(content="No image provided for analysis.")]}


# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("incorrect_question", incorrect_question)
workflow.add_node("analyze_meal_question", analyze_meal_question) 
workflow.add_node("meal_image_scan",meal_image_scan)
workflow.add_node("meal_text_scan",meal_text_scan)
workflow.add_node("barcode",barcode)
workflow.add_node("process_input",process_input)
# Define edges
workflow.set_entry_point("process_input")
workflow.add_edge("process_input","retrieve")
workflow.add_edge("retrieve", "grade_documents")


workflow.add_conditional_edges(
    "grade_documents",
    lambda x: "generate" if x["messages"][-1].content == "Documents are relevant." else "analyze_meal_question",
    {
        "generate": "generate",
        "analyze_meal_question": "analyze_meal_question"
    }
)

workflow.add_conditional_edges(
    "analyze_meal_question",
    lambda x: x["messages"][-1].content.split()[0].lower(),
    {
        "image": "meal_image_scan",
        "text":"meal_text_scan",
        "barcode": "barcode",
        "question": "incorrect_question"
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("incorrect_question", END)
workflow.add_edge("meal_image_scan",END)
workflow.add_edge("meal_text_scan",END)
workflow.add_edge("barcode",END)

# Compile the graph
graph = workflow.compile()

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Amway Chatbot! You can ask questions about products, nutrition, and more.").send()


async def handle_image(message):
    image_input = None

    # Check if there are elements in the message
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.Image):
                try:
                    # Get the image path from the element
                    image_path = element.path
                    if image_path and os.path.exists(image_path):
                        # Read the image data from the file path
                        with open(image_path, 'rb') as f:
                            image_bytes = f.read()
                        
                        # Create a temporary filename for saving the image if needed
                        temp_filename = f"temp_image_{int(time.time())}.jpg"
                        
                        # Optionally save the image to a new temporary file
                        with open(temp_filename, 'wb') as f:
                            f.write(image_bytes)
                        
                        # Create an ImageInput object
                        image_input = ImageInput(filename=temp_filename)
                        print(f"Successfully processed image: {temp_filename}")
                        
                        # For debugging, print image details
                        print(f"Image path: {image_path}")
                        print(f"Image size: {len(image_bytes)} bytes")
                    else:
                        print(f"Image path does not exist: {image_path}")
                except Exception as img_error:
                    print(f"Error processing image: {str(img_error)}")
                break

    return image_input

@cl.on_message
async def main(message: cl.Message):
    # Initialize message elements
    elements = []
    print(cl.Message)
    # Process any uploaded images
    image_input = None

    image_input = await handle_image(message)
    if image_input:
        # You can now process or use the image_input as needed
        print(f"Image saved at: {image_input.filename}")
    else:
        print("No image was uploaded or processed.")
    
    # Create user input
    user_input = UserInput(
        text=message.content,
        image=image_input
    )
    print("user input is:",user_input)

    # Prepare initial state
    initial_state = {
        "user_input": user_input
    }

    #     print("\n---\n")


    # Create a message showing processing
    msg = cl.Message(content="Processing your request...")
    await msg.send()

    try:
        # Run the graph
        for output in graph.stream(initial_state):
            for key, value in output.items():
                if 'messages' in value and value['messages']:
                    latest_message = value['messages'][-1].content
                    print(latest_message)
                    # Update the message content
                    msg.content = latest_message
                    await msg.update() 
                    #await msg.edit(content=latest_message)
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        msg.content = error_message
        await msg.update()
        #await msg.edit(content=error_message)

    finally:
        # Clean up any temporary files if needed
        if image_input and image_input.filename.startswith('temp_image_'):
            import os
            try:
                os.remove(image_input.filename)
            except:
                pass

if __name__ == "__main__":
    cl.run()
