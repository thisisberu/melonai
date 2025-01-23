# includes/graph.py
from typing import Dict, List, Literal, Optional
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

class AgentState(BaseModel):
    messages: List[BaseMessage]
    retrieved_docs: Optional[str] = None
    user_input: Dict


def create_workflow_graph(tool_manager):
    # Initialize state graph
    workflow = StateGraph(AgentState)
    
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
