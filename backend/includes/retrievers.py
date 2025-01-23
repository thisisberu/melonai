from langchain_google_community import VertexAIMultiTurnSearchRetriever
from langchain.tools import create_retriever_tool
import google.auth

from dotenv import load_dotenv 
load_dotenv() 

credentials, project_id = google.auth.default() 
PROJECT_ID = "amw-dna-coe-working-ds-dev"
REGION = "us-central1"
CREDENTIALS = credentials
LOCATION_ID = "global"
DATA_STORE_ID = "amway-articles_1727879500677" 

DATA_STORE_ID2 = "who-blog-content_1727226322567"
DATA_STORE_ID3 = "amway-insider_1717140077416"

DATA_STORE_ID4= "core-plan-document-of-record_1728023514692" 
LOCATION_ID4 = "us" 

def setup_retrievers():
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
    return tools 
