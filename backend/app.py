
import logging
import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Dict, Optional, List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from includes.models import HealthCheckResponse
from includes.chatbot import process_chatbot_request
from  includes.chatbot import chat_manager_handler
import shutil
from typing import Annotated
import aiofiles
from pathlib import Path
from includes.register_process import process_registration_request
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Health + Wellness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatbotRequest(BaseModel):
    text: str
    file: UploadFile
    
class ChatbotResponse(BaseModel):
    response: str
 

class ChatbotResponse2(BaseModel):
    response: str
    user_info: Optional[Dict] = None
    

# Ensure absolute path to temp_uploads directory
UPLOAD_DIR = Path(os.path.abspath("temp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to temporary directory and return the path."""
    try:
        # Create a unique filename to avoid conflicts
        file_path = UPLOAD_DIR / upload_file.filename

        # Debug: Print file details
        print(f"Uploading file: {upload_file.filename}")
        print(f"Content type: {upload_file.content_type}")

        # Read the file content
        content = await upload_file.read()
        
        # Debug: Check content size
        print(f"File content size: {len(content)} bytes")

        # Ensure content is not empty
        if not content:
            raise ValueError("No file content received")

        # Save the file
        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(content)
        
        # Verify file size after writing
        written_file_size = file_path.stat().st_size
        print(f"Written file size: {written_file_size} bytes")

        return file_path

    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
@app.post("/chatbot/", response_model=ChatbotResponse)
async def chatbot_endpoint(
    text: Annotated[str, Form()],
    user_id: Annotated[str, Form()], 
    file: Annotated[UploadFile | None, File()] = None
):
    print(f"Received text: {text}")
    
    filename = None
    file_size = None
    
    if file:
        # Get file information
        # content = await file.read()
        filename = file.filename
        # file_size = len(content)
        print(f"Received file: {filename} with size: {file_size} bytes")
        
        file_path = await save_upload_file(file)
        print(f"Saved file to: {(file_path)}")

        image_wrapper = (str(file_path))

        # Reset the file pointer for future reads if needed
        await file.seek(0)
        try:
            response = process_chatbot_request(text,user_id,image_wrapper)
            file_path.unlink(missing_ok=True)
            return ChatbotResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        print("as file is not there:")
        try:
            response = process_chatbot_request(text,user_id)
            return ChatbotResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/newchat/")
async def new_chat_endpoint(user_id:  Annotated[str, Form()]) -> dict:
    """
    Endpoint to reset the global chat manager.
    Creates a new manager instance, effectively starting a fresh conversation.
    """
    try:
        # Create a new manager
        print("user id is",user_id)
        new_manager = chat_manager_handler.create_new_manager(user_id)
        print("new_manager is",new_manager)
        return {
            "status": "success",
            "message": "New chat session initialized",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create new chat session: {str(e)}")


@app.post("/register/", response_model=ChatbotResponse2)
async def registration_endpoint(
    text: Annotated[str, Form()],
    user_id: Annotated[str, Form()],
    file: Annotated[UploadFile | None, File()] = None
):
    """
    Endpoint for handling registration process interactions.
    Supports both text input and file uploads.
    """
    print(f"Received registration text: {text}")
    
    file_path = None
    
    try:
        if file:
            # Handle file upload if present
            file_path = await save_upload_file(file)
            print(f"Saved registration file to: {file_path}")
            await file.seek(0)
            # Process the registration request with the file
            response = process_registration_request("xyz",user_id, str(file_path))
            
            # Clean up the temporary file
            file_path.unlink(missing_ok=True)
            
        else:
            # Process the registration request without a file
            response = process_registration_request(text, user_id)
        # Handle both string and dictionary responses
        print("typeof resposne",type(response))
        if isinstance(response, dict):
            print("final response",response)
            return ChatbotResponse2(
                response=response["response"],
                user_info=response.get("user_info")
            )
           
        return ChatbotResponse2(response=response)
        
    except Exception as e:
        # Clean up file if there was an error
        if file_path:
            file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


# Clean up function to remove old temporary files
@app.on_event("startup")
async def startup_event():
    """Clean up any old temporary files on startup."""
    for file in UPLOAD_DIR.glob("*"):
        file.unlink(missing_ok=True)


def main():
    """
    This is the main function that runs the application.
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

if __name__ == "__main__":
    main()