# %%
# Standard library imports
import os
import ast 
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Literal, Annotated, Sequence
from datetime import datetime
from IPython.display import Image, Markdown
import asyncio
from rich import print
from google.cloud import bigquery
import pandas as pd 

# Third-party imports
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import print
from sentence_transformers import SentenceTransformer
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain.schema import Document
from typing import List, Dict, Union, Optional, Annotated, Literal, Sequence
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
import numpy as np
import json
from pathlib import Path
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from typing import Dict

import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from google.cloud import translate_v3
import langid

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
LOCATION = (
    "global"  # The location for the Translation API (use a specific region if needed)
)


# Base models
class ImageInput(BaseModel):
    """Represents an uploaded image."""

    filename: str


class UserInput(BaseModel):
    """Represents user input, which can include text and/or an image."""

    text: str
    image: Optional[ImageInput] = None


class AgentState(Dict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_docs: str
    user_input: UserInput
    corrected_question: Optional[str] = None
    has_image: bool = False
    image_path: Optional[str] = None
    user_id: str
    retrieved_products: str
    intents: List[
        str
    ] = []  # Can include: "health", "meal_plate", "product_recommendation"
    agent_responses: Dict[str, str] = {}  # Store responses from different agents
    active_agents: List[str] = []
    intent_queries: Dict[str, List[str]] = {}  # Store sub-queries for each intent
    final_answer: Dict[str,List[str]] = {} 

class EnhancedConversationManager:
    def __init__(self, project_id: str, data_stores: List[Dict[str, str]]):
        """Initialize with multiple data stores."""
        self.project_id = project_id
        self.data_stores = data_stores
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now().isoformat()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.current_context = []
        self.used_sources = []
        # Initialize question history as instance variable
        self.question_history = []
        self.answer_history = []
        self.conversation_history = []
        self.language = ""
        self.conditions = []
        self.products = "" 
        # Single initialization of shared components
        self._initialize_shared_components()
        # Initialize workflow once
        self.initialize_workflow()

    def _initialize_shared_components(self):
        """Initialize API clients and conversations for all data stores at once."""
        print("Starting unified initialization")
        self.clients = {}
        self.conversations = {}

        # Group data stores by location to minimize client creation
        location_grouped_stores = {}
        for store in self.data_stores:
            location = store["location"]
            if location not in location_grouped_stores:
                location_grouped_stores[location] = []
            location_grouped_stores[location].append(store)

        # Create one client per location
        for location, stores in location_grouped_stores.items():
            client_options = (
                ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
                if location != "global"
                else None
            )

            client = discoveryengine.ConversationalSearchServiceClient(
                client_options=client_options
            )

            # Create conversations for all stores in this location
            for store in stores:
                store_id = store["data_store_id"]
                self.clients[store_id] = client

                # Create conversation for this store
                conversation = client.create_conversation(
                    parent=client.data_store_path(
                        project=self.project_id,
                        location=store["location"],
                        data_store=store_id,
                    ),
                    conversation=discoveryengine.Conversation(),
                )
                self.conversations[store_id] = conversation

        print("Completed unified initialization")

    def initialize_workflow(self):
        # Initialize workflow
        self.workflow = StateGraph(AgentState)

        # Add nodes
        self.workflow.add_node("process_input", process_input)
        self.workflow.add_node("initial_agent", initial_agent)
        self.workflow.add_node("intent_agent", intent_agent)
        self.workflow.add_node("process_parallel", process_parallel)
        self.workflow.add_node("generate", generate)
        # self.workflow.add_node("retrieve",retrieve)
        # self.workflow.add_node("meal_plate", analyze_meal_question)
        # self.workflow.add_node("product_recommendation", recommend_products)

        # Define edges
        self.workflow.set_entry_point("process_input")
        self.workflow.add_edge("process_input", "initial_agent")
        self.workflow.add_edge("initial_agent", "intent_agent")
        self.workflow.add_edge("intent_agent", "process_parallel")
        self.workflow.add_edge("process_parallel", "generate")

        self.workflow.add_edge("generate", END)

        # Compile graph
        self.graph = self.workflow.compile()
        # display(Image(self.graph.get_graph().draw_mermaid_png()))

    def add_question_to_history(self, question: str):
        """Add a new question to the history."""
        self.question_history.append(question)

    def get_recent_questions(self, limit: int = 1) -> List[str]:
        """Get the most recent questions, default last 20."""
        return self.question_history[-limit:]

    def process_question(self, user_input: UserInput, user_id: str):
        """Process a question with optional image input."""

        
        inputs = {
            "user_input": user_input,
            "image_processed": False,  # Add flag to track image processing
            "user_id": user_id,
        }

        # Process through workflow
        final_response = ""
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                if isinstance(value, dict) and "messages" in value:
                    messages = value["messages"]
                    if messages:
                        final_response = messages[-1].content
                        # final_answer = messages[-1]
        
        return final_response


# Helper functions
def get_token(auth_url, client_id, client_secret):
    auth_data = {
        "grant_type": "client_credentials",
        "client_secret": client_secret,
        "client_id": client_id,
    }
    response = requests.post(auth_url, json=auth_data)
    # print("-----response from get token----",response.json())
    response.raise_for_status()
    return response.json().get("access_token")


def evaluate_api_response(state, api_response: Dict, user_question: str) -> str:
    # print("-----api_response-----",api_response) 
    # model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    model = ChatGroq(temperature=0,model= "llama-3.3-70b-versatile")
    user_id = state["user_id"]
    manager = chat_manager_handler.get_current_manager(user_id)

    # Split the analysis into two prompts for more focused, concise responses
    question_prompt = PromptTemplate(
        template="""Answer the user's question based on this meal analysis: {meal_analysis}. 

User Question: {user_question}
Provide a concise, conversational response that directly addresses their question. Keep it brief.""",
        input_variables=["meal_analysis", "user_question"],
    )

    gut_health_prompt = PromptTemplate(
        template="""Based on your previous answer, analyze the gut health impact of the mentioned foods using these categories:
GUT-POSITIVE FOODS (Should make up 60% of daily calories):
        - Vegetables and greens: spinach, lettuce, broccoli, cauliflower, onions, asparagus, kale, tomatoes, squash
        - Fruits and berries: apples, oranges, strawberries, kiwis, blueberries, pears, peaches, bananas
        - Legumes, nuts and seeds: lentils, chickpeas, beans, pumpkin seeds, peanuts, sunflower seeds
        - Whole grain pasta, bread, cereal and tortillas
        - Lean protein: chicken breast, turkey, tofu, lean ground beef, eggs, salmon, tuna, trout, shrimp
        - Cultured and fermented foods: kimchi, sauerkraut, kefir, yogurt with live bacteria
        - Healthy fats: olive oil, avocados
        - Herbs and spices
        - Hydrating beverages: water, herbal tea
 
        GUT-NEUTRAL FOODS (Up to 25% of daily calories):
        - Caffeine (limit to 2 cups of coffee per day)
        - Red meats (limit to 2 servings per week)
        - Sweetened and whole dairy: sour cream, cheese, butter
        - Foods with high natural sugar content: honey, grapes, overripe bananas
 
        GUT-NEGATIVE FOODS (Should be limited to 15% of daily calories):
        - Fast foods: fries, burgers, chips, convenience meals, palm oil
        - Processed and fatty meat: bacon, ham, deli meats, salami, sausage, hot dogs, lamb, steak, pork
        - Refined carbs: white bread, biscuits, white pasta, tortillas, white rice, corn products
        - Sugary and artificially sweetened beverages
        - Products with added sugar: candies, cookies, cakes, pastries, ice cream
        - Alcohol

Previous Response: {previous_response}

Provide a brief, friendly analysis that:
1. Now identify which category (gut-positive, neutral, or negative) each food item falls into.Keep it brief. 
2. If any gut-negative foods are identified, suggest healthier gut-positive alternatives that are similar in nature or can satisfy the same craving.
        For example: If pizza is mentioned (gut-negative due to refined carbs), suggest alternatives like:
        - Whole grain pizza crust with vegetable toppings
        - Cauliflower crust pizza with lean protein
        Only suggest alternatives if there are reasonable gut-positive substitutions available. Keep it brief.

Keep it conversational, like you're chatting with a friend. Use emoji sparingly if appropriate.""",
        input_variables=["previous_response"],
    )

    # Create two separate chains
    question_chain = question_prompt | model | StrOutputParser()
    gut_health_chain = gut_health_prompt | model | StrOutputParser()

    # Execute chains sequentially
    initial_response = question_chain.invoke(
        {"meal_analysis": str(api_response), "user_question": user_question}
    )

    gut_health_response = gut_health_chain.invoke(
        {"previous_response": initial_response}
    )

    # Combine responses with appropriate spacing
    final_response = f"{initial_response}\n\n{gut_health_response}"

    manager.answer_history.append(final_response)
    return final_response


def translate_text(state, text, language_code) -> translate_v3.TranslationServiceClient:
    client = translate_v3.TranslationServiceClient()
    parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    print(parent)
    user_id = state["user_id"]
    manager = chat_manager_handler.get_current_manager(user_id)
    # Translate text from English to chosen language
    # Supported mime types: # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        contents=[text],
        target_language_code=language_code,
        parent=parent,
        mime_type="text/plain",
    )
    current_translated_output = ""
    # Display the translation for each input text provided
    for translation in response.translations:
        # print(f"Translated text: {translation.translated_text}")
        manager.language = translation.detected_language_code
        current_translated_output += translation.translated_text

    return current_translated_output


# Main workflow nodes
def process_input(state):
    """Process the initial user input with improved image handling."""
    print("\n---PROCESS INPUT---")
    user_input = state["user_input"]
    print(f"Raw input text: {user_input.text}")

    if user_input.image and not state.get("image_processed", False):
        print(f"Image attached: {user_input.image.filename}")
        content = f"{user_input.text}\n[Attached image: {user_input.image.filename}]"
        state["image_processed"] = True
    else:
        content = user_input.text

    return {
        "messages": [HumanMessage(content=content)],
        "user_input": user_input,
        "image_processed": state.get("image_processed", False),
    }



def initial_agent(state: Dict) -> Dict:
    print("\n---INITIAL AGENT---")
    current_question = state["user_input"].text
    print(f"Input question: {current_question}")

    user_id = state["user_id"]
    manager = chat_manager_handler.get_current_manager(user_id)

    cur_lang = langid.classify(current_question)[0]

    if cur_lang == "en":
        print("Its english already")
        manager.language = "en"
    else:
        current_question = translate_text(state, current_question, "en")
        print("translated_question: ", current_question)

    has_image = state["user_input"].image is not None
    if has_image:
        manager.add_question_to_history(current_question)
        return state
    try:
        if "error" in state:
            return state

        # Initialize the conversation memory with a window size of k=5
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", k=1, return_messages=True
        )

        # Add recent questions to memory
        recent_questions = manager.get_recent_questions()
        reversed_questions = list(reversed(recent_questions))

        # Add recent answers to memory
        recent_answers = manager.answer_history[-1:]
        reversed_answers = list(reversed(recent_answers))

        # Save context for past questions and answers in memory
        for question, answer in zip(reversed_questions, reversed_answers):
            memory.save_context(
                {"input": question}, {"output": answer}
            )  # Save both question and answer

        print("inside memory ", memory)

        prompt_template = PromptTemplate(
            input_variables=["user_input", "chat_history"],
            template=(
                "You are an intelligent assistant designed to process user questions with precise context awareness. "
                "IMPORTANT CONTEXT: 'ABO' or 'ABOs' refers to Amway Business Owner(s). Never change or reinterpret these terms.\n\n"
                "Follow these steps:\n\n"
                "1. First, analyze if the user's question is standalone:\n"
                "   - Can it be fully understood without any context?\n"
                "   - Does it contain all necessary information?\n"
                "   - Is it grammatically complete?\n\n"
                "2. If YES to all above (standalone question):\n"
                "   - Return the question exactly as provided\n"
                "   - DO NOT add any context from chat history\n"
                "   - Only fix obvious grammatical errors if any\n\n"
                "3. If NO to any above (context-dependent question):\n"
                "   - Reference ONLY the immediately preceding interaction:\n"
                "   Chat history: {chat_history}\n"
                "   - Add minimal necessary context to make the question clear\n"
                "   - Maintain the user's original intent and tone\n"
                "   - Focus on pronouns, references, and implicit subjects\n\n"
                "User's question: {user_input}\n\n"
                "Refined question:"
            ),
        )

        # Set up the language model with a specified temperature and model type
        model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
        
        # chain = LLMChain(llm=model, prompt=prompt_template)
        
        chain = prompt_template | model
        
        # Load conversation history from memory
        chat_history = memory.load_memory_variables({})["chat_history"]

        # Invoke the chain to generate a refined question
        result = chain.invoke(
            {
                "user_input": current_question,
                "chat_history": chat_history,  # Pass the loaded chat history
            }
        )
        # Store the corrected question in the state - handle AIMessage result
        state["corrected_question"] = result.content
        
        # Log the current refined question to history
        manager.add_question_to_history(result.content)

        return state

    except Exception as e:
        print(f"Error in initial_agent: {str(e)}")
        state["error"] = str(e)
        return state

# Add new intent detection function
def detect_intent(question: str, has_image: bool) -> List[str]:
    """Determine which agents should handle the question."""
    
    # question = state["corrected_question"]
    # has_image = state["has_image"]
    class IntentClassification(BaseModel):
        """Classification of user question intents."""

        health_intent: bool = Field(
            description="Question relates to health, nutrition,wellness advice or releated to diseases"
        )
        meal_plate_intent: bool = Field(
            description="Question about specific meal, food item, or plate"
        )
        product_intent: bool = Field(
            description="Question asking for product recommendations"
        )
        incorrect_question: bool = Field(
            description="Question is not about health, nutrition, wellness,related to diseases, meal plate, or product recommendation"
        )
        reasoning: str = Field(description="Explanation for the classification")

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm_with_tool = model.with_structured_output(IntentClassification)

    prompt = PromptTemplate(
        template="""Analyze the following question and determine its intents. A question can have multiple intents.

Question: {question}
Image Present: {has_image}

Determine if the question matches any of these intents:

1. MEAL PLATE INTENT (especially important if image present):
    (Priority over health intent if about specific food):
    ANY question about a specific food item or meal should ONLY be classified as meal_plate_intent, NOT health_intent.
    This includes:
    Determine if the question is about a CONCRETE meal, food plate, or named food item.
    It should be marked as a meal question if ANY of these conditions are met:
 
    1. EXPLICIT FOOD REFERENCES:
       - Question asks about a specific, concrete meal or food item
       - Examples: "What's in this sandwich?", "Is this pasta healthy?", "How many calories in this apple?"
 
    2. IMAGE CONTEXT INDICATORS:
       - Question uses demonstrative pronouns ("this", "that") implying something visible
       - Question seeks judgment about edibility or quality ("good to eat", "safe to eat", "looks okay")
       - Question asks about appearance or condition ("does this look right", "is this done")
       - The user_input indicates an image was uploaded
       Examples:
       - "Is this good to eat?"
       - "Does this look right?"
       - "Can I eat this?"
 
    3. IMMEDIATE FOOD DECISIONS:
       - Question asks for judgment about consuming specific food
       - Examples: "Should I eat this?", "Is this safe?", "Good to consume?"
 
    Do NOT mark as a meal question if it's about:
    1. General meal concepts or timing (e.g., "When should I eat breakfast?")
    2. General meal types or categories (e.g., "What is a balanced breakfast?")
    3. Dietary patterns or habits (e.g., "Should I skip breakfast?")
    4. Nutritional concepts (e.g., "Are carbs bad?")
    5. Meal planning or preparation in general
    6. General questions about meal timing
    7. Impact of meals on health in general
 
    Key Distinction: A meal question can be identified either by explicit food references OR by contextual clues indicating
    the user is referring to a specific food item/image, even if not directly stated.

2. HEALTH INTENT (Only for general health topics):
   ONLY for general health questions NOT about specific foods:
   - Questions about nutrition, health impacts, wellness
   - General dietary advice
   - Health benefits or risks
   - Nutritional information requests
   - General wellness principles
   - Disease prevention/management
   - Dietary patterns (not specific meals)
   - General nutritional guidelines

3. PRODUCT RECOMMENDATION INTENT:
   - Expanded Detection Criteria:

     1. HEALTH PRODUCT-SPECIFIC TRIGGERS:
        - Questions specifically about health/wellness products or supplements:
          * "What supplements should I take for..."
          * "Which products are good for..."
          * "Recommend something for..."
          * "Best supplements/products for..."
        
        - Health conditions requiring product solutions:
          * Sleep Health -> sleep supplements
          * Bone Health -> calcium supplements
          * Brain Health -> cognitive supplements
          * Joint Health -> joint support products
          * Vision Health -> eye health products
          * Heart Health -> heart supplements
          * Digestive Health -> probiotics, digestive aids
          * Immunity Health -> immune boosters
          * Energy -> energy supplements
          * Weight Management -> weight management products

     2. PRODUCT-SPECIFIC LANGUAGE:
        Must include explicit or implicit references to:
          * Supplements
          * Vitamins
          * Nutritional products
          * Health aids
          * Wellness products
          * Natural remedies
          * Dietary supplements

     3. EXCLUDE BUSINESS/SALES QUERIES:
        Do NOT classify as product intent if question involves:
          * Business opportunity
          * Sales plans
          * Commission structures
          * Marketing strategies
          * Distributor relationships
          * Business metrics
          * Sales techniques
          * Revenue/income questions
          * Partner programs
          * Business training
          * Leadership levels
          * Recruitment
          * Business meetings
          * Sales targets
          * Market expansion

     4. INTENT VALIDATION:
        Question must satisfy BOTH:
          a) Focus on health/wellness products
          b) Seek product recommendation or information
        
        Examples:
        ✓ "What supplements are good for joint pain?"
        ✓ "Recommend products for better sleep"
        ✗ "How does the Amway compensation plan work?"
        ✗ "Tips for growing my business"

   Detection Strategy:  
   - Look for explicit product needs or implied product solutions
   - Verify health/wellness context
   - Check against business/sales exclusion list
   - Confirm consumer (not business) perspective

4. INCORRECT QUESTION INTENT:
   - Question is not about health, nutrition, wellness,related to diseases, meal plate, or product recommendation

   Detection Strategy:  
   - Analyze entire question context, not just explicit product mentions
   - Use natural language processing to detect underlying product-seeking intent
   - Consider both direct and indirect signals of product recommendation needs

   Exclusion Criteria:
   - Pure informational queries without solution-seeking language
   - Academic or research-oriented questions
   - Purely conceptual health discussions

Consider:
- Multiple intents can be true simultaneously
- Image context significantly increases likelihood of meal plate intent
- Be particularly sensitive to demonstrative pronouns with images

Classify the intents and explain your reasoning.""",
        input_variables=["question", "has_image"],
    )

    chain = prompt | llm_with_tool
    result = chain.invoke({"question": question, "has_image": has_image})

    intents = []
    if result.health_intent:
        intents.append("health")
    if result.meal_plate_intent:
        intents.append("meal_plate")
    if result.product_intent:
        intents.append("product")
    if result.incorrect_question:
        intents.append("others")
    

    return intents


def decompose_complex_query(question: str, has_image: bool) -> List[Dict[str, str]]:
    """
    Intelligently decompose complex queries while avoiding unnecessary breakdown.

    Args:
        question (str): The original user question
        has_image (bool): Whether an image is present

    Returns:
        List[Dict[str, str]]: A list of decomposed query dictionaries
    """

    class QueryDecomposition(BaseModel):
        """Structured output for query decomposition."""

        should_decompose: bool = Field(
            description="Determine if the query needs to be broken down"
        )
        sub_queries: Optional[List[str]] = Field(
            description="List of focused sub-queries if decomposition is needed"
        )
        rationale: Optional[str] = Field(
            description="Explanation for decomposition decision"
        )

    # Configure LLM for structured output
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm_with_tool = model.with_structured_output(QueryDecomposition)

    # Comprehensive decomposition prompt
    decomposition_prompt = PromptTemplate(
        template="""You are an expert query analyzer. Determine if a query truly requires decomposition.
Analyze the query carefully and provide ONLY THE MOST ESSENTIAL sub-queries.

CRITICAL RULES:
1. MINIMIZE the number of sub-queries - each additional query costs time and resources
2. Only include sub-queries that provide UNIQUE, NECESSARY information
3. Avoid redundant or overlapping questions
4. If a single focused query can capture the essence, DO NOT add more

EXAMPLES OF GOOD DECOMPOSITION:

BAD:
Question: "What are herbal remedies for headaches?"
❌ Sub-queries:
   - "What herbal remedies help with headaches?"
   - "How effective are herbal remedies for headaches?"  [Redundant]
   - "What is the history of herbal headache treatments?" [Unnecessary]

GOOD:
✅ Single query is sufficient: "What are herbal remedies for headaches?"

BAD:
Question: "How do diet and exercise affect diabetes management?"
❌ Sub-queries:
   - "How does diet affect diabetes?"
   - "How does exercise affect diabetes?"
   - "What is the relationship between diet and exercise?" [Unnecessary]
   - "What is diabetes management?" [Too basic]

GOOD:
✅ Sub-queries:
   - "How does diet affect diabetes management?"
   - "How does exercise affect diabetes management?"

Remember: Each additional sub-query MUST provide essential, non-redundant information.

Decomposition Criteria:
1. ONLY decompose if the query is GENUINELY COMPLEX
2. Do NOT break down simple, straightforward questions
3. Focus on queries that have multiple distinct aspects or require multifaceted investigation

Guidelines for Decomposition:
- Simple definition or single-concept questions: DO NOT DECOMPOSE
- Questions with multiple independent aspects: DECOMPOSE
- Queries requiring exploration of different dimensions: DECOMPOSE

Examples:
1. "What is gut health?" 
   - Should Decompose: NO
   - Sub-query: None
   - Rationale: Simple definition query

2. "How can I improve my diet to manage diabetes and reduce heart risk?"
   - Should Decompose: YES
   - Potential Sub-queries:
     * "What dietary changes help manage diabetes?"
     * "What nutritional strategies reduce heart disease risk?"

3. "What are the symptoms, causes, and treatments for rheumatoid arthritis?"
   - Should Decompose: YES
   - Potential Sub-queries:
     * "What are the primary symptoms of rheumatoid arthritis?"
     * "What causes rheumatoid arthritis?"
     * "What are the current treatment options for rheumatoid arthritis?"

Original Question: {question}
Image Present: {has_image}

Make a precise determination: Should this query be decomposed?""",
        input_variables=["question", "has_image"],
    )

    # Create processing chain
    chain = decomposition_prompt | llm_with_tool

    try:
        # Analyze query complexity
        decomposition_result = chain.invoke(
            {"question": question, "has_image": has_image}
        )

        # If no decomposition needed, return original query
        if not decomposition_result.should_decompose:
            return [
                {
                    "id": "original_query",
                    "text": question,
                    "original_context": {
                        "full_question": question,
                        "has_image": has_image,
                    },
                }
            ]

        # If decomposition is suggested
        sub_queries = decomposition_result.sub_queries or [question]

        # Prepare detailed sub-queries with context
        processed_sub_queries = []
        for idx, sub_query in enumerate(sub_queries, 1):
            processed_sub_queries.append(
                {
                    "id": f"sub_query_{idx}",
                    "text": sub_query,
                    "original_context": {
                        "full_question": question,
                        "has_image": has_image,
                    },
                }
            )

        # Log decomposition rationale
        print(f"Decomposition Rationale: {decomposition_result.rationale}")

        return processed_sub_queries

    except Exception as e:
        # Fallback mechanism
        print(f"Query decomposition error: {e}")
        return [
            {
                "id": "original_query",
                "text": question,
                "original_context": {"full_question": question, "has_image": has_image},
            }
        ]


def intent_agent(state: Dict) -> Dict:
    """
    Enhanced intent agent with query decomposition and intelligent routing.

    Args:
        state (Dict): Current conversation state

    Returns:
        Dict: Updated conversation state with decomposed queries and agent routing
    """
    print("\n---ADVANCED INTENT AGENT---")
    # print("a")

    # Extract core information
    question = state.get("corrected_question", state["messages"][0].content)
    has_image = state["user_input"].image is not None

    # Decompose query
    decomposed_queries = decompose_complex_query(question, has_image)

    # Prepare state for decomposed processing
    state["decomposed_queries"] = decomposed_queries
    state["active_agents"] = set()
    state["agent_responses"] = {}

    # Process each sub-query for intent detection
    for sub_query in decomposed_queries:
        sub_query_text = sub_query["text"]

        # Detect intents for each sub-query
        intents = detect_intent(sub_query_text, has_image)
        print(f"Sub-query: {sub_query_text}")
        print(f"Detected intents: {intents}")

        # Initialize intent_queries dictionary if it doesn't exist
        if "intent_queries" not in state:
            state["intent_queries"] = {
                "retrieve": [],
                "meal_plate": [],
                "product_recommendation": [],
                "others": [],
            }

        # Add the sub-query to each detected intent's list
        for intent in intents:
            if intent == "health":
                state["intent_queries"]["retrieve"].append(sub_query)
            elif intent == "meal_plate":
                state["intent_queries"]["meal_plate"].append(sub_query)
            elif intent == "product":
                state["intent_queries"]["product_recommendation"].append(sub_query)
            elif intent == "others":
                state["intent_queries"]["others"].append(sub_query)
    return state


from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List


def process_parallel(state: Dict) -> Dict:
    """Process multiple agents in parallel using ThreadPoolExecutor."""

    # Initialize active agents based on non-empty intent queries
    state["active_agents"] = [
        agent for agent, queries in state["intent_queries"].items() if queries
    ]

    def run_agent(agent_name: str, query_text: str) -> Dict:
        """Execute a specific agent based on its name."""

        try:
            if agent_name == "retrieve":
                ans = retrieve(state, query_text)
                return {"retrieve": ans}
            elif agent_name == "meal_plate":
                return {"meal_plate": analyze_meal_question(state, query_text)}
            elif agent_name == "product_recommendation":
                return {"product_recommendation": recommend_products(state, query_text)}
            elif agent_name == "others":
                return {"others": others(state, query_text)}
            return {}
        except Exception as e:
            print(f"Error in {agent_name} agent: {e}")
            return {}

    # Create a list of all agent-query pairs that need to be processed
    agent_query_pairs = []
    for agent, queries in state["intent_queries"].items():
        for query in queries:
            agent_query_pairs.append((agent, query["text"]))

    print("agent_query_pairs", agent_query_pairs)
    # Use ThreadPoolExecutor to process all queries in parallel
    with ThreadPoolExecutor(max_workers=len(agent_query_pairs)) as executor:
        # Submit tasks for each agent-query pair
        future_to_pair = {
            executor.submit(run_agent, agent, query_text): (agent, query_text)
            for agent, query_text in agent_query_pairs
        }

        # Initialize results storage
        results = {"retrieve": [], "meal_plate": [], "product_recommendation": [], "others": []}

        # Collect results
        for future in as_completed(future_to_pair):
            agent, _ = future_to_pair[future]
            try:
                result = future.result()
                # Append the result to the appropriate list
                for agent_name, response in result.items():
                    if response:  # Only append non-empty responses
                        results[agent_name].append(response)
            except Exception as e:
                print(f"Error processing query for {agent}: {e}")

        # Merge results for each agent
        merged_state = state.copy()
        merged_state["agent_responses"] = {}

        # Handle each agent type separately
        if results["retrieve"]:
            # Combine retrieved docs from all retrieve responses
            all_docs = []
            for response in results["retrieve"]:
                if isinstance(response, dict) and "retrieved_docs" in response:
                    all_docs.extend(response["retrieved_docs"])
            merged_state["agent_responses"]["retrieve"] = {"retrieved_docs": all_docs}

        if results["meal_plate"]:
            # Join meal plate responses if they're strings, otherwise keep the last response
            meal_responses = []
            for response in results["meal_plate"]:
                if isinstance(response, str):
                    meal_responses.append(response)
                elif isinstance(response, dict) and "messages" in response:
                    meal_responses.append(response["messages"][0].content)
            merged_state["agent_responses"]["meal_plate"] = (
                "\n".join(meal_responses) if meal_responses else ""
            )

        # if results["product_recommendation"]:
        #     # Combine product recommendations
        #     all_products = []
        #     for response in results["product_recommendation"]:
        #         if isinstance(response, dict) and "retrieved_products" in response:
        #             all_products.extend(response["retrieved_products"])
        #     merged_state["agent_responses"]["product_recommendation"] = {
        #         "retrieved_products": all_products
        #     }
        
        if results["product_recommendation"]:            
            # Process and structure product recommendations
            unique_products = {}
            consultation_insight = ""
            
            for response in results["product_recommendation"]:
                if isinstance(response, dict):
                    # Deduplicate products
                    if "retrieved_products" in response:
                        for product in response["retrieved_products"]:
                            unique_products[product['product_name']] = product
                    
                    # Capture consultation insight
                    if "retrieved_result" in response:
                        consultation_insight = response["retrieved_result"]
            
            # Convert unique products to list
            all_products = list(unique_products.values())
            
            # Structure the merged state with clear differentiation
            merged_state["agent_responses"]["product_recommendation"] = {
                "retrieved_products": all_products,
                "consultation_insight": consultation_insight
            }

        if results["others"]:
            all_docs = []
            for response in results["others"]:
                if isinstance(response, dict) and "others" in response:
                    all_docs.extend(response["others"])
            merged_state["agent_responses"]["others"] = {"others": all_docs}
    
    return merged_state

def get_bigquery_schema(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    
    # Get the full table reference
    table_ref = client.dataset(dataset_id).table(table_id)
    
    # Retrieve the table
    table = client.get_table(table_ref)
    
    schema_details = []
    for field in table.schema:
        schema_details.append({
            'name': field.name,
            'type': field.field_type,
            'mode': field.mode,
        })
    
    return schema_details 

def execute_bigquery_query(query: str) -> str:
    """Execute BigQuery query and return results as formatted string"""
    try:
        client = bigquery.Client()
        query_job = client.query(query)
        results = query_job.result()
        
        # Convert results to DataFrame and then to string
        df = results.to_dataframe()
        if len(df) > 10:  # Limit large results
            df = df.head(10)
        return df.to_string()
    except Exception as e:
        return f"Error executing query: {str(e)}"

def clean_sql_query(text: str) -> str:
    """
    Clean SQL query by removing code block syntax, various SQL tags, backticks,
    prefixes, and unnecessary whitespace while preserving the core SQL query.

    Args:
        text (str): Raw SQL query text that may contain code blocks, tags, and backticks

    Returns:
        str: Cleaned SQL query
    """
    # Step 1: Remove code block syntax and any SQL-related tags
    # This handles variations like ```sql, ```SQL, ```SQLQuery, etc.
    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text = re.sub(block_pattern, r"\1", text, flags=re.DOTALL)

    # Step 2: Handle "SQLQuery:" prefix and similar variations
    # This will match patterns like "SQLQuery:", "SQL Query:", "MySQL:", etc.
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQL)\s*:\s*"
    text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)

    # Step 3: Extract the first SQL statement if there's random text after it
    # Look for a complete SQL statement ending with semicolon
    sql_statement_pattern = r"(SELECT.*?;)"
    sql_match = re.search(sql_statement_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if sql_match:
        text = sql_match.group(1)

    # Step 4: Remove backticks around identifiers
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # Step 5: Normalize whitespace
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Step 6: Preserve newlines for main SQL keywords to maintain readability
    keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
               'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN',
               'OUTER JOIN', 'UNION', 'VALUES', 'INSERT', 'UPDATE', 'DELETE']

    # Case-insensitive replacement for keywords
    pattern = '|'.join(r'\b{}\b'.format(k) for k in keywords)
    text = re.sub(f'({pattern})', r'\n\1', text, flags=re.IGNORECASE)

    # Step 7: Final cleanup
    # Remove leading/trailing whitespace and extra newlines
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n', text)

    return text

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate

def generate_bigquery_query(schema: str, question: str, state: Dict = None) -> str:
    """Generate BigQuery SQL query from natural language question"""
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
    
    # Get user_id/abo_id from state if available
    abo_id = state.get("user_id") if state else None
    
    # Define few-shot examples
    examples = [
        {
            "input": "How does my monthly volume this year compare to last year (higher or lower)?",
            "query": """SELECT 
    CASE 
        WHEN current_year_avg_total_downline_pv_normalized_to_10k > last_year_avg_total_downline_pv_normalized_to_10k 
        THEN 'higher' 
        ELSE 'lower'
    END AS current_year_volume_vs_last_year
FROM `amw-dna-coe-working-ds-dev.data_science.abo_info`
WHERE global_account_id = {abo_id}"""
        },
        {
            "input": "How did my total downline pv from last year compare to two years ago?",
            "query": """SELECT 
    CASE 
        WHEN last_year_avg_total_downline_pv_normalized_to_10k > two_years_ago_avg_total_downline_pv_normalized_to_10k 
        THEN 'higher' 
        ELSE 'lower'
    END AS last_year_volume_vs_two_years_ago
FROM `amw-dna-coe-working-ds-dev.data_science.abo_info`
WHERE global_account_id = {abo_id}"""
        }
    ]

    # Create example prompt template
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}")
    ])

    # Create few-shot prompt template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        input_variables=["input"]
    )

    # Enhanced schema description with detailed column information
    detailed_schema = """
    TABLE COLUMNS AND DESCRIPTIONS:
    
    1. Business Identification Columns:
    - aff_id: Three digit affiliate number that uniquely identifies the affiliate
    - global_account_id: Unique global ABO identifier (calculated by adding 11 zeros to affiliate ID + ABO Number/Account ID)
    
    2. Performance Year Information:
    - base_current_perf_year: The column identifies the current performance year referred to in the data row.
    - current_year_completed_months:The number of months that have been completed in the performance year indicated in the base_current_perf_year_column.
    - last_year_completed_months: This will be 12 unless the ABO started their business last year (one year before the base current year).
    - two_years_ago_completed_months: The number of months the ABO was in business two years before the base current perf year.
    - three_years_ago_completed_months: The number of months the ABO was in business three years before the base current perf year.
    
    3. Downline PV Metrics (Normalized to 10k):
    - current_year_avg_total_downline_pv_normalized_to_10k: The monthly average of total downline pv normalized to 10k for the current perf year as indicated in the base_current_perf_year column.
    - last_year_avg_total_downline_pv_normalized_to_10k: The monthly average of total downline pv normalized to 10k for the perf year 1 year before the perf years as indicated in the base_current_perf_year column.
    - two_years_ago_avg_total_downline_pv_normalized_to_10k: The monthly average of total downline pv normalized to 10k for the perf year 2 years before the perf years as indicated in the base_current_perf_year column.
    - three_years_ago_avg_total_downline_pv_normalized_to_10k:The monthly average of total downline pv normalized to 10k for the perf year 3 years before the perf years as indicated in the base_current_perf_year column.
    
    4. Core Plan Bonus Metrics (USD):
    - current_year_avg_in_market_core_plan_bonus_usd: The monthly average of in market core plan bonuses for the current perf year as indicated in the base_current_perf_year column.
    - last_year_avg_in_market_core_plan_bonus_usd: The monthly average of in market core plan bonuses stated in USD for the perf year 1 year before the perf years as indicated in the base_current_perf_year column.
    - two_years_ago_avg_in_market_core_plan_bonus_usd: The monthly average of in market core plan bonuses stated in USD for the perf year 2 years before the perf years as indicated in the base_current_perf_year column.
    - three_years_ago_avg_in_market_core_plan_bonus_usd: The monthly average of in market core plan bonuses stated in USD for the perf year 3 years before the perf years as indicated in the base_current_perf_year column.
    """

    # Create the main prompt template with few-shot examples and enhanced schema
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analyst who converts natural language questions into BigQuery SQL queries.
        
        IMPORTANT CONTEXT:
        - The table is located at: `amw-dna-coe-working-ds-dev.data_science.abo_info`
        - Current ABO ID (if needed): {abo_id}
        
        {detailed_schema}
        
        Additional Schema Information:
        {schema}
        
        QUERY GUIDELINES:
        1. Always use the full table path: `amw-dna-coe-working-ds-dev.data_science.abo_info`
        2. If the question implies personal data or "my" information, use the ABO ID filter
        3. For general queries, don't include ABO ID filter
        4. Always include appropriate LIMIT clause for large result sets
        5. Use clear column aliases for better readability
        6. Consider the normalization of PV values (normalized to 10k) when comparing volumes
        7. Be aware that bonus values are in USD
        8. Account for completed months when analyzing year-over-year comparisons
        
        Write only the SQL query, nothing else. Ensure it's a valid BigQuery SQL query."""),
        few_shot_prompt,
        ("human", "{question}\nSQLQuery:"),
    ])
    
    # Generate the query using the enhanced prompt
    chain = final_prompt | model | StrOutputParser()
    
    query = chain.invoke({
        "schema": schema,
        "detailed_schema": detailed_schema,
        "question": question,
        "abo_id": abo_id if abo_id else "NULL",
        "input": question  # Required for few-shot template
    })
    
    return clean_sql_query(query)


def others(state: Dict, query_text: str) -> Dict:
    """Handle general queries including database questions"""
    print("\n-----Running query analysis-----")
    project_id = "amw-dna-coe-working-ds-dev"
    dataset_id = "data_science"
    table_id = "abo_info" 

    # First determine if this is a database query
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
    model = ChatGroq(temperature=0, model = "llama-3.3-70b-versatile")
    analyze_prompt = PromptTemplate(
    template="""Determine if this question requires database lookup or should be answered from the sales plan knowledge base.

    Question: {question}

    CLASSIFICATION RULES:

    1. DATABASE QUERIES (Return 'database'):
       Any questions about:
       a) Personal Information:
          - ABO/Affiliate IDs
          - Account details
          - Registration dates
          - Personal status/level
          - Individual qualifications
       
       b) Numerical/Statistical Data:
          - Member counts
          - Performance metrics
          - Sales figures
          - Achievement statistics
       
       c) Relationship Data:
          - Upline/downline information
          - Team structure
          - Sponsor details
       
       d) Transaction/History:
          - Purchase history
          - Commission records
          - Point calculations
          - PV/BV queries

    2. SALES PLAN QUERIES (Return 'other'):
       Any questions about:
       a) Business Understanding:
          - How the compensation plan works
          - Commission structures
          - Bonus calculations
          - Leadership levels
          - Qualification requirements
       
       b) Program Information:
          - Business opportunity
          - Sales plan details
          - Reward programs
          - Recognition systems
       
       c) General Knowledge:
          - Business policies
          - Company procedures
          - Program benefits
          - Growth opportunities

    Examples:
    DATABASE:
    - "What's my ABO ID?" -> 'database'
    - "Show my current PV" -> 'database'
    - "Who is in my downline?" -> 'database'
    - "Check my qualification status" -> 'database'

    SALES PLAN (OTHER):
    - "How do I reach Platinum level?" -> 'other'
    - "Explain the compensation structure" -> 'other'
    - "What are the leadership qualifications?" -> 'other'
    - "Tell me about the bonus program" -> 'other'

    Return ONLY 'database' for data lookup queries,
    return 'other' for sales plan and general information questions.
    """,
    input_variables=["question"]
)
    
    analyze_chain = analyze_prompt | model | StrOutputParser()
    
    query_type = analyze_chain.invoke({"question": query_text}).strip().lower()
    print(f"Detected query type: {query_type}")

    if query_type == 'database':
        try:
            # Get schema
            schema = get_bigquery_schema(project_id, dataset_id, table_id)
            
            # Generate SQL query
            sql_query = generate_bigquery_query(schema, query_text, state)
            print(f"------Generated SQL query:--------/n {sql_query}")
            
            # Execute query
            query_results = execute_bigquery_query(sql_query)
            
            # Generate natural language response
            response_prompt = PromptTemplate(
                template="""Based on the following database query and results, provide a natural language response.
                
                Question: {question}
                SQL Query: {sql_query}
                Results: {results}
                
                Provide a clear, concise explanation of the results in natural language. Don't add unneccasary info just keep it straight and to the point. 
                """,
                input_variables=["question", "sql_query", "results"]
            )
            
            response_chain = response_prompt | model | StrOutputParser()
            natural_response = response_chain.invoke({
                "question": query_text,
                "sql_query": sql_query,
                "results": query_results
            })
            
            return {"others": [{"content": natural_response, "source": "BigQuery Database"}]}
            
        except Exception as e:
            error_msg = f"Error processing database query: {str(e)}"
            return {"others": [{"content": error_msg, "source": "Error"}]}
    else:
        print("-----Processing sales plan query-----")
        # Initialize Discovery Engine client and conversation
        project_id = "amw-dna-coe-working-ds-dev"
        data_stores = [
            {"location": "global", "data_store_id": "abo-sales-plan-hw_1734077229422"},
        ]
        user_id = state["user_id"]
        manager = chat_manager_handler.get_current_manager(user_id)
        
        # Create Discovery Engine client
        client_options = ClientOptions(api_endpoint="global-discoveryengine.googleapis.com")
        client = discoveryengine.ConversationalSearchServiceClient(client_options=client_options)
        
        # Create conversation
        conversation = client.create_conversation(
            parent=client.data_store_path(
                project=project_id,
                location="global",
                data_store=data_stores[0]["data_store_id"],
            ),
            conversation=discoveryengine.Conversation(),
        )

        # Make the search request
        request = discoveryengine.ConverseConversationRequest(
            name=conversation.name,
            query=discoveryengine.TextInput(input=query_text),
            serving_config=client.serving_config_path(
                project=project_id,
                location="global",
                data_store=data_stores[0]["data_store_id"],
                serving_config="default_config",
            ),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                include_citations=True,
            ),
        )

        try:
            response = client.converse_conversation(request)
            results = []

            for result in response.search_results:
                result_data = result.document.derived_struct_data
                content = result_data.get("snippets", [{}])[0].get("snippet", "")
                source = result_data.get("link", "Unknown source")
                title = result_data.get("title", "Untitled")
                results.append({
                    "content": content,
                    "source": source,
                    "title": title,
                })

            ranked_results = rank_documents(manager, results, query_text)
            # print("ranked_results",ranked_results)
            return {"others": ranked_results}
            
        except Exception as e:
            error_msg = f"Error processing sales plan query: {str(e)}"
            return {"others": [{"content": error_msg, "source": "Error"}]}
        

def recommend_products2(state: Dict, query_text: str) -> Dict:
    print("running recommend_products")
    project_id = "amw-dna-coe-working-ds-dev"
    location = "global"
    engine_id = "product-search-app_1732269184635"
    # search_query = state.get("corrected_question", state["messages"][0].content)
    search_query = query_text
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search app serving config
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=10,
        # content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)
    result = next(response.pages)

    search_results = result.results
    all_search_result = []

    for i, result1 in enumerate(search_results, 1):
        result_doc = result1.document.struct_data
        search_result = {}
        search_result["product_name"] = result_doc["product_name"]
        search_result["product_url"] = result_doc["product_url"]

        all_search_result.append(search_result)
    print("all_search_result", all_search_result)
    top_n = 3
    return {
        "retrieved_products": all_search_result[:top_n],
    }

def recommend_products(state: Dict, query_text: str) -> Dict:
    # print("running recommend_products")
    project_id = "amw-dna-coe-working-ds-dev"
    location = "global"
    engine_id = "product-search-app_1732269184635"
    # search_query = state.get("corrected_question", state["messages"][0].content)
    search_query = query_text
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    user_id = state["user_id"]
    manager = chat_manager_handler.get_current_manager(user_id)

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search app serving config
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=10,
        # content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)
    result = next(response.pages)

    search_results = result.results
    all_search_result = []
    
    product_search_result_global = []
    manager.products = ""
    # print("Results from product: ", search_results)
    for i, result1 in enumerate(search_results, 1):
        result_doc = result1.document.struct_data
        search_result = {}
        products_global_data = {}
        search_result["product_name"] = result_doc["product_name"]
        search_result["product_url"] = result_doc["product_url"]
        
        products_global_data["product_name"] = result_doc["product_name"]
        products_global_data["product_description"] = result_doc["product_description"]
        
        product_search_result_global.append(products_global_data)
        all_search_result.append(search_result)
        
    top_n = 3    
    product_search_result_global = product_search_result_global[:top_n]    
    manager.products += str(product_search_result_global)
    
    shopping_result = shopping(state, query_text)
    
    print("Result from shopping Intent to Product one: ", shopping_result)
    
    
    return {
        "retrieved_products": all_search_result[:top_n],
        "retrieved_result": shopping_result
    }

def compute_similarity(query_embedding, doc_embedding):
    """Compute cosine similarity between embeddings."""
    return np.dot(query_embedding, doc_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
    )


def rank_documents(
    manager: EnhancedConversationManager, docs: List[Dict], query: str
) -> List[Dict]:
    """Rank documents based on relevance to query."""
    query_embedding = manager.model.encode([query])[0]

    doc_similarities = []
    for doc in docs:
        content = doc["content"]
        doc_embedding = manager.model.encode([content])[0]
        similarity = compute_similarity(query_embedding, doc_embedding)
        doc_similarities.append((similarity, doc))

    threshold = 0.3
    ranked_docs = [
        doc
        for score, doc in sorted(doc_similarities, key=lambda x: x[0], reverse=True)
        if score > threshold
    ]
    return ranked_docs


def search_data_store(
    manager: EnhancedConversationManager, query: str, store_info: Dict[str, str]
) -> List[Dict]:
    """Search a single data store and return results."""
    client = manager.clients[store_info["data_store_id"]]
    conversation = manager.conversations[store_info["data_store_id"]]
    request = discoveryengine.ConverseConversationRequest(
        name=conversation.name,
        query=discoveryengine.TextInput(input=query),
        serving_config=client.serving_config_path(
            project=manager.project_id,
            location=store_info["location"],
            data_store=store_info["data_store_id"],
            serving_config="default_config",
        ),
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            include_citations=True,
        ),
    )

    response = client.converse_conversation(request)
    results = []

    for result in response.search_results:
        result_data = result.document.derived_struct_data
        content = result_data.get("snippets", [{}])[0].get("snippet", "")
        source = result_data.get("link", "Unknown source")
        title = result_data.get("title", "Untitled")
        results.append(
            {
                "content": content,
                "source": source,
                "title": title,
            }
        )

    return results


def search_and_rank(manager: EnhancedConversationManager, question: str) -> Dict:
    """Process a question and return a comprehensive response."""
    # Search all data stores in parallel
    with ThreadPoolExecutor() as executor:
        all_results = []
        futures = []

        # Create futures with correct arguments
        for store_info in manager.data_stores:
            future = executor.submit(
                search_data_store,
                manager,  # Pass the manager instance
                question,  # Pass the question
                store_info,  # Pass the store info dictionary
            )
            futures.append(future)

        for future in futures:
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Error searching data store: {e}")

    # Rank and filter results
    ranked_results = rank_documents(
        manager, all_results, question
    )  # Pass manager and query
    # print("ranked docs",ranked_results)
    return ranked_results


def retrieve(state: Dict, query_text: str) -> Dict:
    print("\n---RETRIEVE DOCUMENTS---")
    # Get the corrected question from state
    # corrected_question = state.get("corrected_question", state["messages"][0].content)
    corrected_question = query_text
    print(f"Processing question: {corrected_question}")
    user_id = state["user_id"]
    manager = chat_manager_handler.get_current_manager(user_id)

    # Pass the manager instance and corrected question
    manager.current_context = search_and_rank(manager, corrected_question)
    top_n = 8
    # print("manager.current_context",manager.current_context[:top_n])

    state["retrieved_docs"] = manager.current_context[:top_n]
    return {
        "retrieved_docs": manager.current_context[:top_n],
        "corrected_question": corrected_question,  # Maintain the corrected question
    }

def parse_response(response):
    """
    Returns:
       dict: Dictionary containing parsed sections with preserved formatting
    """
    # Initialize the dictionary with empty default values
    result = {"answer": "", "sources": [], "more_topics": []}

    # Split the response into lines
    lines = response.split("\n")

    current_section = "answer"
    for line in lines:
        stripped_line = line.strip()

        # Check for section transitions
        if stripped_line.lower().startswith("sources:"):
            current_section = "sources"
            continue
        elif stripped_line.lower().startswith("would you like to know more about:"):
            current_section = "more_topics"
            continue

        # Process line based on current section
        if current_section == "answer":
            # Preserve original line formatting for answer
            if result["answer"]:
                result["answer"] += "\n" + line
            else:
                result["answer"] = line
        elif current_section == "sources":
            # Handle sources with bullet points or dashes
            if stripped_line.startswith("-") or stripped_line.startswith("•"):
                result["sources"].append(stripped_line[1:].strip())
            elif stripped_line:
                result["sources"].append(stripped_line)
        elif current_section == "more_topics":
            # Handle more topics with bullet points or dashes
            if stripped_line.startswith("-") or stripped_line.startswith("•"):
                result["more_topics"].append(stripped_line[1:].strip())
            elif stripped_line and not stripped_line.lower().startswith(
                "would you like to know more about"
            ):
                result["more_topics"].append(stripped_line)

    # Clean up answer by removing leading/trailing whitespace while preserving internal formatting
    result["answer"] = result["answer"].strip()

    return result

def generate(state: Dict) -> Dict:
    """Generate final response combining all agent outputs with proper formatting."""
    print("\n---GENERATE ANSWER---")
    corrected_question = state.get("corrected_question", state["messages"][0].content)
    user_id = state["user_id"]
    manager = chat_manager_handler.get_current_manager(user_id)
    response = ""
    # Prepare for relevance grading
    class grade(BaseModel):
        binary_score: Literal["yes", "no"]

    # Set up LLM for grading
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm_with_tool = model.with_structured_output(grade)

    # Gather health documents
    health_docs = state.get("agent_responses", {}).get("retrieve", "")
    health_context = ""
    others_context = "" 

    consultation_insight = (
        state.get("agent_responses", {})
        .get("product_recommendation", {})
        .get("consultation_insight", "")
    )

    # Check if health docs are not empty and perform relevance grading
    if (
        health_docs
        and isinstance(health_docs, dict)
        and health_docs.get("retrieved_docs")
    ):
        # Combine documents into context
        docs_context = "\n\n".join(
            [
                f"Document: {doc['content']}\nSource: {doc['source']}"
                for doc in health_docs["retrieved_docs"]
            ]
        )
        health_context = docs_context

    # Get meal response
    meal_response = state.get("agent_responses", {}).get("meal_plate", "")

    others_response = state.get("agent_responses", {}).get("others", "")
    # print("others_response",others_response)
    if others_response and isinstance(others_response, dict) and others_response.get("others") and others_response["others"] != []:
        others_context = "\n\n".join(
            [
                f"Document: {doc['content']}\nSource: {doc['source']}"
                for doc in others_response["others"]
            ]
        )

    # print("others_context",others_context)# Extract product recommendations
    
    product_recommendations = (
        state.get("agent_responses", {})
        .get("product_recommendation", {})
        .get("retrieved_products", [])
    )

    # Only proceed if at least one context is non-empty
    full_response = ""
    if health_context or meal_response or others_context or consultation_insight:
        merge_prompt = PromptTemplate(
            template="""You are a friendly and knowledgeable chatbot having a conversation with the user. Synthesize information from multiple sources into a coherent, well-structured response.

**AVAILABLE INFORMATION:**
Health Information:
{health_context}        

Meal Analysis:
{meal_response}

Product Information:
{consultation_insight}

Database Results or Sales Plan Rag Results:
{others_context}

**Database HANDLING RULES:**

1. **Database Results Processing:**
   - ONLY use exact values/data points present in the database results
   - DO NOT interpolate, extrapolate, or infer trends unless explicitly shown in the data
   - If asked about data not present in results, clearly state "This information is not available in the current data"
   - Format numerical data exactly as shown in database results
   - Present database information using direct quotes or exact numbers 

2. **Data Restrictions:**
   - NO hypothetical scenarios based on the data
   - NO predictions or forecasting
   - NO combining database results with external knowledge
   - ONLY discuss relationships explicitly present in the data
   - If data seems incomplete or inconsistent, state this explicitly

**RESPONSE GUIDELINES:**
 
1. **Conversational Style:**
   - Use a friendly, casual tone while maintaining professionalism
   - Keep responses brief and to-the-point (2-3 short paragraphs max)
   - Break complex information into digestible chunks
   - Use simple, everyday language
   - Add brief conversational transitions when needed
 
2. **Health & Safety Rules:**
 
    FIRST: Check if question contains ANY of:
    - Medical conditions (diabetes, arthritis, cancer, etc.)
    - Diet/nutrition topics
    - Treatments/medications
    - Symptoms or health concerns
    - Emergency situations
 
    THEN:
    - Use valid source URLs in plain text
    - Include ALL applicable disclaimers from below:
 
    **Required Disclaimers:**
 
    FOR ANY DIET/NUTRITION CONTENT:
    *Please note: For personalized dietary advice and meal planning tailored to your health needs, consult a registered dietitian.*
 
    FOR ANY MEDICAL CONDITIONS:
    *Important: This information is general education only. For medical advice specific to your [condition_name], please consult your healthcare provider.*
 
    FOR ANY TREATMENTS/MEDICATIONS:
    *Medical Disclaimer: Never start or modify any medication/treatment without consulting a qualified healthcare professional.*
 
    FOR EMERGENCY TOPICS:
    *EMERGENCY: If you're experiencing a medical emergency, contact emergency services immediately.*
 
    **Disclaimer Rules:**
    - Format in italics using asterisks
    - Include BOTH medical and dietary disclaimers for condition-specific diet questions
    - Always include condition name in medical disclaimer
    - NOTE: Never skip disclaimers for health-related content

3. **Answer Structure:**
   - Start with a direct, concise answer
   - Use bullet points for multiple items
   - Keep explanations brief but clear
   - Include source attribution naturally in conversation
 
4. **Sources and Follow-ups:**
   - End with "Sources:" followed by cited sources in plain text and complete URLs
   - Add "Would you like to know more about:" followed by 2 brief, relevant follow-up topics (not questions)
   - Keep both sections short and professional
 
5. **Formatting:**
   - Use simple markdown for readability
   - Include paragraph breaks for easy reading
   - Use bullet points sparingly and only when listing items
 
6. **Source Compliance:**
   - Stick to information from provided context
   - Don't create additional information
   - Attribute sources conversationally

 
  
Your response should end with:
 
Sources:
[List sources here in plain text]
 
Would you like to know more about:
• [Topic 1]
• [Topic 2]

Question: {corrected_question}

Remember to be friendly and conversational while providing accurate, source-based information. Keep responses concise. Ensure smooth transitions between different types of information.""",
            input_variables=["health_context", "meal_response","others_context", "consultation_insight","corrected_question"],
        )

        generate_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = merge_prompt | generate_model | StrOutputParser()

        # Generate the main response
        response = chain.invoke(
            {
                "health_context": health_context,
                "meal_response": meal_response,
                "others_context": others_context,
                "consultation_insight": consultation_insight,
                "corrected_question": corrected_question,
            }
        )
        
    # Format product recommendations in markdown
    if product_recommendations:
        product_section = "\n\n---\n\n**Recommended Products:**\n"
        for product in product_recommendations:
            product_section += (
                f"- [{product['product_name']}]({product['product_url']})\n"
            )

        # Append product recommendations to the response
        
        full_response = response + product_section
    else:
        full_response = response
    
    print(full_response)
    

    # Parse and store response
    if response!="":
        result = parse_response(response)
    else:
        result = {"answer": "", "sources": [], "more_topics": []} 
    
    translated_result = {}
    if manager.language == "en":
        pass
    else:
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Handle single string translation for 'answer'
            future_to_key = {
                executor.submit(
                    translate_text, state, result["answer"], manager.language
                ): "answer"
            }

            # Handle list translation for 'more_topics'
            if isinstance(result["more_topics"], list):
                # Translate each topic in the list
                more_topics_futures = [
                    executor.submit(translate_text, state, topic, manager.language)
                    for topic in result["more_topics"]
                ]
                future_to_key.update(
                    {
                        future: f"more_topics_{i}"
                        for i, future in enumerate(more_topics_futures)
                    }
                )
            else:
                # If it's a single string, translate it like 'answer'
                future_to_key[
                    executor.submit(
                        translate_text, result["more_topics"], manager.language
                    )
                ] = "more_topics"
            result["more_topics"] = []
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    if key == "answer":
                        result["answer"] = future.result()
                    elif key == "more_topics":
                        result["more_topics"] = future.result()
                    elif key.startswith("more_topics_"):
                        # Reconstruct the list of translated topics
                        result["more_topics"].append(future.result())
                except Exception as exc:
                    print(f"Translation for {key} generated an exception: {exc}")
                    if key == "answer":
                        result["answer"] = result["answer"]
                    elif key == "more_topics":
                        result["more_topics"] = result["more_topics"]
                    elif key.startswith("more_topics_"):
                        if "more_topics" not in translated_result:
                            result["more_topics"] = []
                        translated_result["more_topics"].append(
                            result["more_topics"][int(key.split("_")[-1])]
                        )
    
    if product_recommendations:
        result["products"] = product_section
    manager.answer_history.append(result["answer"])
    manager.conversation_history.append(result)

    return {
        "messages": [HumanMessage(content=str(result))],
        "corrected_question": corrected_question, 
    }

def shopping(state: Dict, query_text: str) -> Dict:
    print("\n---SHOPPING AGENT---")
    corrected_question = query_text
    user_id = state["user_id"]
    manager = chat_manager_handler.get_current_manager(user_id)
    
    prompt_template = PromptTemplate(
        input_variables=["user_input", "product_description"],
        template="""PRODUCT CONSULTATION PROTOCOL

        CORE DIRECTIVE:
            - Analyze user query: {user_input}
            - Reference available product data: {product_description}
            - Provide a direct, precise answer
            - Use only factual product information
            - Avoid unnecessary details
            - Answer exactly what is asked
            - Be concise and to the point

        RESPONSE GUIDELINES:
            1. Understand the specific question
            2. Extract relevant product information
            3. Deliver a clear, straightforward response
            4. If no direct answer is possible, state limitations clearly

        CRITICAL CONSTRAINTS:
            - Maximum brevity
            - Absolute precision
            - Direct addressing of the query
            - No speculative or generalized information""",
    )
    # Set up the language model with a specified temperature and model type
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    chain = LLMChain(llm=model, prompt=prompt_template)

    # Invoke the chain to generate a refined question
    result = chain.invoke(
        {
            "user_input": corrected_question,
            "product_description": manager.products,
        }
    )
    
    output_text = result["text"]
    # print(result)
    return output_text



def incorrect_question(state):
    """
    Handle case when the question is deemed incorrect or irrelevant.
    """
    print("---INCORRECT QUESTION---")
    return {
        "messages": [
            HumanMessage(
                content="The question appears to be incorrect or irrelevant to the available information."
            )
        ]
    }


def analyze_meal_question(state, question: str) -> Dict:
    print("---ANALYZE MEAL QUESTION---",question)

    class QuestionAnalysis(BaseModel):
        """Analysis result for the user's question."""

        is_meal_related: bool = Field(
            description="Whether the question is related to meal analysis"
        )
        api_call_type: str = Field(
            description="Type of API call needed: 'image_scan', 'text_query', 'barcode_scan', or 'none'"
        )
        reasoning: str = Field(description="Explanation for the decision")

    # model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    model = ChatGroq(temperature=0,model = "llama-3.3-70b-versatile") 
    llm_with_tool = model.with_structured_output(QuestionAnalysis)

    prompt = PromptTemplate(
        template="""Analyze the following user question and context:

        User Question: {question}
        Has Image Attachment: {has_image}
        Image Path (if any): {image_path}

        Determine if the question is related to meal plate analysis and what type of API call is needed.
        The question is meal-related if it asks about food items, nutritional information, or involves analyzing a meal image or barcode.

        Possible API call types:
        1. 'image_scan': If there's an image attachment and the question involves analyzing the food/meal in the image
        2. 'text_query': If the question is about specific food items or nutritional information without image analysis
        3. 'barcode_scan': If there's an image attachment and the question involves scanning a barcode
        4. 'none': If the question is not related to meal analysis

        Consider:
        - If there's an image attachment, prioritize image analysis unless explicitly asking for barcode scanning
        - Text queries should only be used when no image is provided
        - The presence of words like "this", "in the image", "this meal" with an image strongly suggests image scanning

        Return your analysis with clear reasoning for the decision.
        """,
        input_variables=["question", "has_image", "image_path"],
    )

    chain = prompt | llm_with_tool

    # question = state.get("corrected_question", state["messages"][0].content)
    user_input = state["user_input"]
    image_path = user_input.image.filename if user_input.image else None
    has_image = user_input.image is not None

    # Perform analysis
    analysis_result = chain.invoke(
        {"question": question, "has_image": has_image, "image_path": image_path}
    )
    print("-----analysis_result-----",analysis_result) 
    # If not meal-related, return early
    if not analysis_result.is_meal_related:
        return {
            "messages": [
                HumanMessage(content="Question is not related to meal analysis.")
            ],
            "question": question,
        }

    # Perform API call based on analysis
    try:
        # Get OAuth Token
        # TOKEN = get_token(
        #     "https://api-dv.amwayglobal.com/rest/oauth2/v1/token",
        #     "3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M",
        #     "9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu",
        # )
        TOKEN = get_token(
            "https://api-dv.amwayglobal.com/rest/oauth2/v1/token",
            "3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M",
            "9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu",
        )

        print("-----token-----",TOKEN)
        # Prepare common headers
        headers_o = {
            "Authorization": f"Bearer {TOKEN}",
            "x-hw-program": "mg_testing",
            "x-abold": "mg_abo",
            "x-mealtime": "",
            "x-genai-vendor": "openai",
            "x-country-code": "mg_testing",
        }

        # Perform API call based on analysis type
        if analysis_result.api_call_type == "image_scan" and has_image:
            print("-----image scan-----")
            headers_o = {
            "Authorization": f"Bearer {TOKEN}",
            "x-hw-program": "mg_testing",
            "x-abold": "mg_abo",
            "x-mealtime": "",
            "x-genai-vendor": "openai",
            "x-country-code": "mg_testing",
        }
            with open(user_input.image.filename, "rb") as img_file:
                files = {"meal_image": img_file}
                response = requests.post(
                    "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/meal-scan",
                    headers=headers_o,
                    files=files,
                    data={},
                )
            print("-----response-----",response.json()) 
        elif analysis_result.api_call_type == "text_query":
            print("-----text query-----")
            data = {"meal_description": question}
            response = requests.post(
                "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/meal-scan",
                headers=headers_o,
                files={},
                data=data,
            )
            print("-----response-----",response.json())
        elif analysis_result.api_call_type == "barcode_scan" and has_image:
            with open(image_path, "rb") as img_file:
                upc_file = {"image": img_file}
                response = requests.post(
                    "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/upc",
                    files=upc_file,
                    headers=headers_o,
                    data={},
                )
        else:
            return {
                "messages": [
                    HumanMessage(content="No applicable meal analysis method found.")
                ],
                "question": question,
            }

        # Evaluate API response
        answer = evaluate_api_response(state, response.json(), question)
        return {"messages": [HumanMessage(content=answer)]}

    except Exception as e:
        return {
            "messages": [HumanMessage(content=f"Error in meal analysis: {str(e)}")],
            "question": question,
        }


def meal_image_scan(state):
    """
    Handle case when the question is deemed incorrect or irrelevant.
    """
    print("---Scanning the image---")
    user_input = state["user_input"]
    user_question = state.get("corrected_question", state["messages"][0].content)
    if user_input.image:
        TOKEN = get_token(
            "https://api-dv.amwayglobal.com/rest/oauth2/v1/token",
            "3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M",
            "9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu",
        )
        headers_o = {
            "Authorization": f"Bearer {TOKEN}",
            "x-hw-program": "mg_testing",
            "x-abold": "mg_abo",
            "x-mealtime": "",
            "x-genai-vendor": "openai",
            "x-country-code": "mg_testing",
        }
        with open(user_input.image.filename, "rb") as img_file:
            files = {"meal_image": img_file}
            response = requests.post(
                "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/meal-scan",
                headers=headers_o,
                files=files,
                data={},
            )

            # print("the response:",response.json())
            answer = evaluate_api_response(state, response.json(), user_question)
            return {"messages": [HumanMessage(content=answer)]}
    else:
        return {"messages": [HumanMessage(content="No image provided for analysis.")]}


def meal_text_scan(state):
    """
    Handle case when the question is deemed incorrect or irrelevant.
    """
    print("---Text scan --")
    user_input = state["user_input"]
    user_question = state.get("corrected_question", state["messages"][0].content)
    data = {"meal_description": user_question}
    TOKEN = get_token(
        "https://api-dv.amwayglobal.com/rest/oauth2/v1/token",
        "3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M",
        "9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu",
    )
    headers_o = {
        "Authorization": f"Bearer {TOKEN}",
        "x-hw-program": "mg_testing",
        "x-abold": "mg_abo",
        "x-mealtime": "",
        "x-genai-vendor": "openai",
        "x-country-code": "mg_testing",
    }
    response = requests.post(
        "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/meal-scan",
        headers=headers_o,
        files={},
        data=data,
    )
    answer = evaluate_api_response(state, response.json(), user_question)
    return {"messages": [HumanMessage(content=answer)]}


def barcode(state):
    """
    Handle case when the question is deemed incorrect or irrelevant.
    """
    print("---Barcode---")
    user_input = state["user_input"]
    user_question = state.get("corrected_question", state["messages"][0].content)
    if user_input.image:
        TOKEN = get_token(
            "https://api-dv.amwayglobal.com/rest/oauth2/v1/token",
            "3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M",
            "9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu",
        )
        headers_o = {
            "Authorization": f"Bearer {TOKEN}",
            "x-hw-program": "mg_testing",
            "x-abold": "mg_abo",
            "x-mealtime": "",
            "x-genai-vendor": "openai",
        }
        with open(user_input.image.filename, "rb") as img_file:
            upc_file = {"image": img_file}
            response = requests.post(
                "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/upc",
                files=upc_file,
                headers=headers_o,
                data={},
            )
            answer = evaluate_api_response(state, response.json(), user_question)
            return {"messages": [HumanMessage(content=answer)]}
    else:
        return {"messages": [HumanMessage(content="No image provided for analysis.")]}


PROJECT_ID = "amw-dna-coe-working-ds-dev"
data_stores = [
    {"location": "global", "data_store_id": "amway-articles_1727879500677"},
    {"location": "global", "data_store_id": "who-blog-unchunked_1728571550919"},
    {"location": "global", "data_store_id": "demo-fbs-store_1729253761309"},
]


class ChatManagerHandler:
    def __init__(self):
        self._managers = {}  # Dict to store managers for each user
        # self._lock = Lock()
        self._cleanup_interval = (
            24 * 3600
        )  # Cleanup interval in seconds (e.g., 24 hours)

    def create_new_manager(self, user_id: str) -> "EnhancedConversationManager":
        """
        Create a new manager for a specific user
        """
        manager = EnhancedConversationManager(PROJECT_ID, data_stores)
        self._managers[user_id] = {
            "manager": manager,
        }
        print("created new manager")
        return manager

    def get_current_manager(self, user_id: str) -> "EnhancedConversationManager":
        """
        Get the manager for a specific user, creating one if it doesn't exist
        """
        if user_id not in self._managers:
            print("user id not found")
            return self.create_new_manager(user_id)

        return self._managers[user_id]["manager"]


# Global instance
chat_manager_handler = ChatManagerHandler()
manager = None


def process_chatbot_request(
    text: str, user_id: str, image_filename: Optional[str] = None
) -> tuple[str, str]:
    """ """
    print("img name is", image_filename)
    image_input = ImageInput(filename=image_filename) if image_filename else None
    print(image_input)
    user_input = UserInput(text=text, image=image_input)
    print(user_input)
    initial_state = {
        "user_input": user_input,
        "messages": [],
        "retrieved_docs": "",
    }

    final_output = ""
    print("user id in chatbot api is", user_id)
    manager = chat_manager_handler.get_current_manager(user_id)
    final_output = manager.process_question(user_input, user_id)
    return final_output






# %%
