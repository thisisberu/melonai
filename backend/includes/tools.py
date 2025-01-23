# includes/tools.py
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from typing import List, Union, Dict, Any
from pydantic import BaseModel

class TokenManager:
    @staticmethod
    def get_token(auth_url: str, client_id: str, client_secret: str) -> str:
        auth_data = {
            "grant_type": "client_credentials",
            "client_secret": client_secret,
            "client_id": client_id
        }
        response = requests.post(auth_url, json=auth_data)
        response.raise_for_status()
        return response.json().get("access_token")

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_similarity(self, query_embedding, doc_embedding):
        return np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )

    def get_content(self, doc: Union[Document, str]) -> str:
        if isinstance(doc, Document):
            return doc.page_content
        return doc

    def rank_documents(self, docs: List[Union[Document, str]], query: str) -> List[Union[Document, str]]:
        query_embedding = self.model.encode([query])[0]
        
        doc_similarities = []
        for doc in docs:
            content = self.get_content(doc)
            doc_embedding = self.model.encode([content])[0]
            similarity = self.compute_similarity(query_embedding, doc_embedding)
            doc_similarities.append((similarity, doc))
        
        return [doc for _, doc in sorted(doc_similarities, key=lambda x: x[0], reverse=True)]

class MealAnalyzer:
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.auth_url = "https://api-dv.amwayglobal.com/rest/oauth2/v1/token"
        self.client_id = "3hmyXKbHlA0ZLJ1Zjtg4G1X0l4srn0jIolK7pzB4EqiqBb1M"
        self.client_secret = "9Trey6amtSaRifSzU1HM2UlirkSLojkBCa0xWA51nUkyFeoGFFfVKWEuGdV8pNbu"
        self.base_headers = {
            "x-hw-program": "mg_testing",
            "x-abold": "mg_abo",
            "x-mealtime": "",
            "x-genai-vendor": "openai",
            "x-country-code": "mg_testing"
        }

    def get_headers(self):
        token = self.token_manager.get_token(self.auth_url, self.client_id, self.client_secret)
        return {**self.base_headers, "Authorization": f"Bearer {token}"}

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as img_file:
            files = {'meal_image': img_file}
            response = requests.post(
                "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/meal-scan",
                headers=self.get_headers(),
                files=files
            )
            return response.json()

    def analyze_text(self, text: str) -> Dict[str, Any]:
        response = requests.post(
            "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/meal-scan",
            headers=self.get_headers(),
            data={'meal_description': text}
        )
        return response.json()

    def analyze_barcode(self, image_path: str) -> Dict[str, Any]:
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            response = requests.post(
                "https://api-dv.amwayglobal.com/v1/health-wellbeing/mealanalyzer/upc",
                headers=self.get_headers(),
                files=files
            )
            return response.json()

class ToolManager:
    def __init__(self, retriever_tools: List, meal_analyzer: MealAnalyzer, document_processor: DocumentProcessor):
        self.retriever_tools = retriever_tools
        self.meal_analyzer = meal_analyzer
        self.document_processor = document_processor

    def get_tools(self):
        return self.retriever_tools

def create_tool_manager(retriever_tools: List) -> ToolManager:
    token_manager = TokenManager()
    meal_analyzer = MealAnalyzer(token_manager)
    document_processor = DocumentProcessor()
    return ToolManager(retriever_tools, meal_analyzer, document_processor)