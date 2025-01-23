"""
This file contains the models for the API requests and responses.
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Sequence, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class HealthCheckResponse(BaseModel):
    """
    Health check response model.
    """

    status: str = Field(default="ok", description="The status of the service")
    timestamp: str = Field(description="The timestamp of the response")

class ChatResponse(BaseModel):
    response: str 

