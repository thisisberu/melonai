from vertexai.generative_models import GenerativeModel
from google.cloud import vision
from typing import Dict, Optional
import datetime
import difflib
import json
import os
import re
from PIL import Image

# Initialize the Gemini model
model = GenerativeModel("gemini-pro")

class RegistrationMemory:
    def __init__(self):
        self.history = []
        self.user_info = {
            # Basic details
            "full_name": None,
            "date_of_birth": None,
            "gender": None,
            # Identity details
            "national_id": None,
            "identity_proof": None,
            # Final Verification
            "id_proof_image": None
        }
        self.current_section = "basic"
        self.sections = ["basic", "identity", "verification"]
        self.section_fields = {
            "basic": ["full_name", "date_of_birth", "gender"],
            "identity": ["national_id", "identity_proof"],
            "verification": ["id_proof_image"]
        }
        self.section_names = {
            "basic": "Basic Information",
            "identity": "Identity Verification",
            "verification": "Final Verification"
        }

        self.field_descriptions = {
            "full_name": "Full Legal Name",
            "date_of_birth": "Date of Birth",
            "gender": "Gender",
            "national_id": "National ID Number",
            "identity_proof": "ID Type",
            "id_proof_image": "Identity Proof Document"
        }
    
    def add_interaction(self, role: str, content: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append({"timestamp": timestamp, "role": role, "content": content})
        self.save_conversation()

    def get_context(self):
        return self.history[-5:]

    def update_user_info(self, field: str, value: str):
        self.user_info[field] = value
        self.save_progress()

    def is_section_complete(self) -> bool:
        return all(
            self.user_info[field] is not None
            for field in self.section_fields[self.current_section]
        )

    def move_to_next_section(self) -> bool:
        if self.is_section_complete():
            current_index = self.sections.index(self.current_section)
            if current_index < len(self.sections) - 1:
                self.current_section = self.sections[current_index + 1]
                return True
        return False

    def get_current_field(self) -> str:
        for field in self.section_fields[self.current_section]:
            if self.user_info[field] is None:
                return field
        return None

    def get_progress_summary(self) -> str:
        summary = "\nRegistration Progress:\n"
        for section in self.sections:
            fields = self.section_fields[section]
            completed_fields = sum(1 for f in fields if self.user_info[f] is not None)
            total_fields = len(fields)

            if section == self.current_section:
                summary += f"\n➤ {self.section_names[section]} ({completed_fields}/{total_fields})\n"
            else:
                summary += f"\n  {self.section_names[section]} ({completed_fields}/{total_fields})\n"

            for field in fields:
                value = self.user_info[field]
                status = "✓" if value else "○"
                display_value = value if value else "Not provided"

                if field in ["national_id", "card_number"]:
                    display_value = "*" * 8 + display_value[-4:] if value else "Not provided"
                elif field in ["profile_picture", "id_proof_image"]:
                    display_value = "Uploaded" if value else "Not uploaded"

                summary += f"   {status} {self.field_descriptions[field]}: {display_value}\n"

        return summary

    def save_progress(self):
        progress_data = {
            "user_info": self.user_info,
            "current_section": self.current_section,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs("registration_data", exist_ok=True)
        with open("registration_data/progress.json", "w") as f:
            json.dump(progress_data, f, indent=2)

    def save_conversation(self):
        os.makedirs("registration_data", exist_ok=True)
        with open("registration_data/conversation_history.txt", "w", encoding="utf-8") as f:
            for interaction in self.history:
                timestamp = interaction["timestamp"]
                role = interaction["role"]
                content = interaction["content"]
                f.write(f"[{timestamp}] {role.upper()}: {content}\n\n")


def is_valid_image(file_path: str) -> bool:
    try:
        with Image.open(file_path) as img:
            return img.format.lower() in ['jpeg', 'jpg', 'png']
    except:
        return False

def validate_image_file(file_path: str, field_type: str) -> tuple[bool, Optional[str]]:
    """Validate image file format and size."""
    if not os.path.exists(file_path):
        return False, "File not found"
    
    if not is_valid_image(file_path):
        return False, "Invalid image format. Please upload JPG, JPEG, or PNG files only"

    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    if file_size > 5:  # 5MB limit
        return False, "File size too large. Please upload images under 5MB"

    return True, file_path

def is_english_text(text: str) -> bool:
    english_chars = sum(1 for c in text if ord('A') <= ord(c) <= ord('Z') or ord('a') <= ord(c) <= ord('z'))
    total_chars = len(text.replace(" ", ""))
    return total_chars > 0 and english_chars / total_chars > 0.5

def extract_id_card_info(image_path: str) -> Dict[str, Optional[str]]:
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        return {
            "name": None,
            "date_of_birth": None,
            "gender": None,
            "id_number": None,
            "raw_text": "",
        }

    raw_text = texts[0].description
    extracted_info = {
        "name": None,
        "date_of_birth": None,
        "gender": None,
        "id_number": None,
        "raw_text": raw_text,
    }

    lines = raw_text.split("\n")
    patterns = {
        "date": r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b",
        "year": r"(?:YoB|जन्म वर्ष).*?(\d{4})",
        "id_number": r"\b\d{4}\s*\d{4}\s*\d{4}\b",
        "gender": r"\b(Male|Female|पुरुष|महिला)\b",
        "english_name": r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b",
    }

    # Extract ID number
    for line in lines:
        id_matches = re.findall(patterns["id_number"], line.replace(" ", ""))
        if id_matches:
            id_num = id_matches[0]
            extracted_info["id_number"] = "".join([id_num[i:i+4] for i in range(0, len(id_num), 4)])
            break

    # Find government line index
    govt_index = -1
    for i, line in enumerate(lines):
        if re.search(r"Government\s+of\s+India", line, re.IGNORECASE):
            govt_index = i
            break

    def is_valid_name(line: str) -> bool:
        invalid_keywords = [
            "GOVERNMENT", "आधार", "DOB", "जन्म", "YoB", "Gender", "सरकार",
            "सत्यमेव", "INDIA", "भारत", "SAMPLE", "MALE", "FEMALE", "Download",
            "Issue", "VID", "Date"
        ]
        if any(keyword.lower() in line.lower() for keyword in invalid_keywords):
            return False
        if re.search(patterns["id_number"], line.replace(" ", "")):
            return False
        if not is_english_text(line):
            return False
        return True

    # Look for name
    if govt_index != -1:
        search_range = list(range(max(0, govt_index - 3), govt_index)) + \
                      list(range(govt_index + 1, min(len(lines), govt_index + 4)))
        for i in search_range:
            line = lines[i]
            if is_valid_name(line):
                words = line.split()
                english_words = []
                for word in words:
                    if re.match(r"^[A-Z][a-zA-Z]+$", word) and any(c.islower() for c in word):
                        english_words.append(word)
                if english_words:
                    extracted_info["name"] = " ".join(english_words)
                    break

    # Extract date of birth
    for line in lines:
        if "DOB" in line or "जन्म" in line:
            dob_matches = re.findall(patterns["date"], line)
            if dob_matches:
                extracted_info["date_of_birth"] = dob_matches[0]
                break
        yob_matches = re.findall(patterns["year"], line)
        if yob_matches:
            extracted_info["date_of_birth"] = f"Year: {yob_matches[0]}"
            break

    # Extract gender
    for line in lines:
        if "MALE" in line.upper() or "पुरुष" in line:
            extracted_info["gender"] = "Male"
            break
        elif "FEMALE" in line.upper() or "महिला" in line:
            extracted_info["gender"] = "Female"
            break

    return extracted_info

def verify_identity(memory: RegistrationMemory, extracted_info: Dict[str, Optional[str]]) -> tuple[bool, str]:
    # Compare extracted information with user-provided information
    mismatches = []
    
    # Check name similarity using difflib
    
    if extracted_info["name"] and memory.user_info["full_name"]:
        name_similarity = difflib.SequenceMatcher(None, 
            extracted_info["name"].lower(), 
            memory.user_info["full_name"].lower()
        ).ratio()
        if name_similarity < 0.8:  # 80% similarity threshold
            mismatches.append("name")

    # Compare gender
    print("extracted info", extracted_info["gender"])
    print("user info", memory.user_info["gender"])
    if extracted_info["gender"] and memory.user_info["gender"]:
        if extracted_info["gender"].lower() != memory.user_info["gender"].lower():
            mismatches.append("gender")

    # Compare date of birth
    print("extracted info", extracted_info["date_of_birth"])
    print("user info", memory.user_info["date_of_birth"])
    if extracted_info["date_of_birth"] and memory.user_info["date_of_birth"]:
        # Handle different date formats
        extracted_dob = re.sub(r'[/-]', '', extracted_info["date_of_birth"])
        user_dob = re.sub(r'[/-]', '', memory.user_info["date_of_birth"])
        if extracted_dob != user_dob:
            mismatches.append("date of birth")

    # Compare ID number
    print("extracted info", extracted_info["id_number"])
    print("user info", memory.user_info["national_id"])
    if extracted_info["id_number"] and memory.user_info["national_id"]:
        if extracted_info["id_number"].replace(" ", "") != memory.user_info["national_id"].replace(" ", ""):
            mismatches.append("ID number")

    if mismatches:
        return False, f"Verification failed. Mismatches found in: {', '.join(mismatches)}"
    return True, "Identity verification successful!"


section_completion_messages = {
    "basic": """Perfect! I've recorded all your basic information. Let's proceed with identity verification.""",
    "identity": """Thank you for providing your identity details. Now, let's complete the final verification step."""
}

def generate_question(memory: RegistrationMemory) -> str:
    section_intros = {
        "basic": "I'm excited to help you start your journey! Let's begin with some basic information.",
        "identity": "You're doing great! For security, we need to verify your identity.",
        "verification": "To complete your registration, please provide your identity proof document for verification."
    }

    user_name = memory.user_info["full_name"]
    personal_greeting = f"{user_name.split()[0]}, " if user_name else ""

    field_questions = {
        "full_name": "What is your full legal name as it appears on your government ID?",
        "date_of_birth": f"{personal_greeting}could you please share your date of birth (DD/MM/YYYY)?",
        "gender": f"{personal_greeting}what gender should we record for your registration?",
        "national_id": f"{personal_greeting}please provide your national ID number.",
        "identity_proof": f"{personal_greeting}what type of government-issued ID are you using?",
        "id_proof_image": f"{personal_greeting}please provide a clear image of your government-issued ID (JPG, JPEG, or PNG format, max 5MB).",
    }
def clean_response_text(response_text: str) -> str:
    """Clean the response text by removing markdown code blocks."""
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def parse_response(json_str: str) -> Optional[Dict]:
    """Parse the JSON response and validate its structure."""
    try:
        data = json.loads(json_str)

        # Handle query type responses
        if data.get("type") == "query":
            return {"type": "query", "content": data.get("content")}

        # Handle information type responses
        if data.get("type") == "information":
            field = data.get("field")
            content = data.get("content")

            # Ensure both field and content are present
            if field and content is not None:
                return {"type": "information", "field": field, "content": content}
    except json.JSONDecodeError:
        pass
    return None

def extract_info(user_input: str, memory: RegistrationMemory) -> Dict:
    current_fields = memory.section_fields[memory.current_section]
    missing_fields = [
        field for field in current_fields if memory.user_info[field] is None
    ]
    current_field = memory.get_current_field()

    # Handle image file uploads
    if current_field in ["profile_picture", "id_proof_image"]:
        if os.path.exists(user_input):
            is_valid, result = validate_image_file(user_input, current_field)
            if is_valid:
                return {
                    "type": "information",
                    "field": current_field,
                    "content": result,
                }
            return {"type": "invalid", "content": result}
        return {"type": "invalid", "content": "Please provide a valid file path"}

    prompt = f"""
    Given the following context:
    - Current section: {memory.current_section}
    - Field we're looking for: {missing_fields}
    - User's input: "{user_input}"
    - Previous conversation: {json.dumps(memory.get_context(), indent=2)}

    Extract the relevant information and standardize specific fields:

    1. For date_of_birth:
       - Convert any date format to DD-MM-YYYY
       - Example: "1990-01-15" or "15-01-1990" should become "15/01/1990"
    2. For gender:
       - Standardize to exactly one of: "Male", "Female", "Others", "Prefer not to say"
       - Example: "m" or "male" becomes "Male"
       - Example: "f" becomes "Female"
       - Example: anything unclear becomes "Prefer not to say"

    If the user is asking a question or making a general statement, return {{"type": "query", "content": user_input}}

    For information extraction, return {{"type": "information", "field": "field_name", "content": "extracted_value"}}

    Return only the JSON object, no other text or explanation.
    """

    response = model.generate_content(prompt)
    print("\nResponse from model:", response.text)
    cleaned_response = clean_response_text(response.text)
    print("\nCleaned response:", cleaned_response)
    result = parse_response(cleaned_response)
    print("\nParsed result:", result)

    if result["type"] == "information":
        field = result["field"]
        content = result["content"]

        # Apply field-specific validation
        if field == "mobile_number":
            is_valid, validated_content = validate_mobile(content)
            if not is_valid:
                return {"type": "invalid", "content": validated_content}
            content = validated_content

        elif field == "email":
            is_valid, validated_content = validate_email(content)
            if not is_valid:
                return {"type": "invalid", "content": validated_content}
            content = validated_content

        elif field == "card_number":
            is_valid, validated_content = validate_card_number(content)
            if not is_valid:
                return {"type": "invalid", "content": validated_content}
            content = validated_content

        return {"type": "information", "field": field, "content": content}

    return result

def handle_query(query: str, memory: RegistrationMemory) -> str:
    prompt = f"""
    You are an AI Registration Assistant for Amway Business Owner registration.
    User query: "{query}"
    Current section: {memory.section_names[memory.current_section]}
    Provide a helpful, concise response.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def validate_mobile(number: str) -> tuple[bool, str]:
    cleaned = re.sub(r"[^0-9]", "", number)
    if len(cleaned) != 10:
        return False, "Mobile number must be 10 digits"
    return True, cleaned

def validate_email(email: str) -> tuple[bool, str]:
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        return False, "Invalid email format"
    return True, email.lower()

def validate_card_number(card_number: str) -> tuple[bool, str]:
    cleaned = re.sub(r"[^0-9]", "", card_number)
    if len(cleaned) != 16:
        return False, "Card number must be 16 digits"
    return True, cleaned

def chatbot():
    memory = RegistrationMemory()
    print("\n🌟 Welcome to Amway Business Owner Registration! 🌟")
    memory.add_interaction(
        "AI Assistant", "Welcome to Amway Business Owner Registration!"
    )

    while True:
        # Generate and display the current question
        question = generate_question(memory)
        print("\nAI Assistant:", question)
        memory.add_interaction("AI Assistant", question)

        # Get user input
        user_input = input("\n👤 You: ").strip()
        if not user_input:
            continue

        memory.add_interaction("You", user_input)

        if user_input.lower() in ["quit", "exit"]:
            print(
                "\nAI Assistant: Thank you for your interest in Amway. You can resume your registration anytime!"
            )
            break

        # Process user input
        result = extract_info(user_input, memory)

        if result["type"] == "query":
            response = handle_query(result["content"], memory)
            print("\nAI Assistant:", response)
            memory.add_interaction("AI Assistant", response)
            continue

        elif result["type"] == "information":
            memory.update_user_info(result["field"], result["content"])

            # Check if this section is complete
            if memory.is_section_complete():
                if memory.current_section == "verification":
                    # Process ID verification
                    extracted_info = extract_id_card_info(
                        memory.user_info["id_proof_image"]
                    )
                    is_verified, verification_message = verify_identity(
                        memory, extracted_info
                    )

                    if is_verified:
                        completion_message = """
                        🎉 Congratulations! Your registration is complete and your identity has been verified.
                        Welcome to the Amway family! 
                        
                        Next steps:
                        1. You'll receive your welcome kit within 5-7 business days
                        2. Your sponsor will contact you shortly to help you get started
                        3. You can now log in to your Amway account and place your first order
                        
                        We're excited to have you join our community of successful entrepreneurs!
                        """
                        print("\nAI Assistant:", completion_message)
                        memory.add_interaction("AI Assistant", completion_message)
                        break
                    else:
                        print("\nAI Assistant:", verification_message)
                        memory.add_interaction("AI Assistant", verification_message)
                        # Reset verification section to try again
                        memory.user_info["id_proof_image"] = None
                        continue

                elif memory.current_section == "payment":
                    # Move to verification section
                    memory.move_to_next_section()
                    next_question = generate_question(memory)
                    print("\nAI Assistant:", section_completion_messages["payment"])
                    print(next_question)
                    memory.add_interaction(
                        "AI Assistant", section_completion_messages["payment"]
                    )
                    memory.add_interaction("AI Assistant", next_question)
                else:
                    # Handle completion of other sections
                    old_section = memory.current_section
                    memory.move_to_next_section()
                    completion_message = section_completion_messages[old_section]
                    next_question = generate_question(memory)
                    print("\nAI Assistant:", f"{completion_message}\n\n{next_question}")
                    memory.add_interaction(
                        "AI Assistant", f"{completion_message}\n\n{next_question}"
                    )
        else:
            print(
                "\nAI Assistant:",
                f"I apologize, but {result['content']}. Could you please try again?",
            )
            memory.add_interaction(
                "AI Assistant",
                f"I apologize, but {result['content']}. Could you please try again?",
            )

# ... existing imports ...

class ChatManagerHandler:
    def __init__(self):
        self._managers = {}  # Dict to store RegistrationMemory instances for each user
        self._cleanup_interval = 24 * 3600  # Cleanup interval in seconds

    def create_new_manager(self, user_id: str) -> "RegistrationMemory":
        """Create a new RegistrationMemory instance for a specific user"""
        manager = RegistrationMemory()
        self._managers[user_id] = {
            "manager": manager,
        }
        print(f"Created new registration manager for user {user_id}")
        return manager

    def get_current_manager(self, user_id: str) -> "RegistrationMemory":
        """Get the RegistrationMemory for a specific user, creating one if it doesn't exist"""
        if user_id not in self._managers:
            print(f"Creating new registration for user {user_id}")
            return self.create_new_manager(user_id)
        return self._managers[user_id]["manager"]

# Global instance
chat_manager_handler = ChatManagerHandler()

def process_registration_request(text: str, user_id: str, image_filename: Optional[str] = None) -> str:
    """Process a single registration interaction"""
    memory = chat_manager_handler.get_current_manager(user_id)
    
    # if not text:
    #     return "Please provide some input."

    memory.add_interaction("You", text)

    # Handle image uploads
    if image_filename and os.path.exists(image_filename):
        print("file_name",image_filename)
        text = image_filename  # Use the image path as the input

    # Process user input
    result = extract_info(text, memory)

    if result["type"] == "query":
        response = handle_query(result["content"], memory)
        memory.add_interaction("AI Assistant", response)
        return response

    elif result["type"] == "information":
        memory.update_user_info(result["field"], result["content"])

        # Check if this section is complete
        if memory.is_section_complete():
            if memory.current_section == "verification":
                # Process ID verification
                extracted_info = extract_id_card_info(memory.user_info["id_proof_image"])
                is_verified, verification_message = verify_identity(memory, extracted_info)

                if is_verified:
                    completion_message = """
                    🎉 Congratulations! Your registration is complete and your identity has been verified.
                    Welcome to the Amway family!
                    
                    Next steps:
                    1. You'll receive your welcome kit within 5-7 business days
                    2. Your sponsor will contact you shortly to help you get started
                    3. You can now log in to your Amway account and place your first order
                    
                    We're excited to have you join our community of successful entrepreneurs!
                    """
                    memory.add_interaction("AI Assistant", completion_message)
                    return completion_message
                else:
                    memory.add_interaction("AI Assistant", verification_message)
                    memory.user_info["id_proof_image"] = None
                    return verification_message

            # Handle section completion
            old_section = memory.current_section
            memory.move_to_next_section()
            completion_message = section_completion_messages.get(old_section, "")
            next_question = generate_question(memory)
            response = f"{completion_message}\n\n{next_question}"
            memory.add_interaction("AI Assistant", response)
            return response

        # If section not complete, get next question
        next_question = generate_question(memory)
        memory.add_interaction("AI Assistant", next_question)
        # print("\nAI Assistant:", next_question)
        return next_question

    else:
        error_message = f"I apologize, but {result['content']}. Could you please try again?"
        memory.add_interaction("AI Assistant", error_message)
        return error_message

