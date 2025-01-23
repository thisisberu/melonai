# melonai

AI based Agents for all your health and wellbeing's needs.

````
███╗   ███╗███████╗██╗      ██████╗ ███╗   ██╗     █████╗ ██╗
████╗ ████║██╔════╝██║     ██╔═══██╗████╗  ██║    ██╔══██╗██║
██╔████╔██║█████╗  ██║     ██║   ██║██╔██╗ ██║    ███████║██║
██║╚██╔╝██║██╔══╝  ██║     ██║   ██║██║╚██╗██║    ██╔══██║██║
██║ ╚═╝ ██║███████╗███████╗╚██████╔╝██║ ╚████║    ██║  ██║██║
╚═╝     ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝  ╚═╝╚═╝


```
 ███▄ ▄███▓▓█████  ██▓     ▒█████   ███▄    █     ▄▄▄       ██▓
▓██▒▀█▀ ██▒▓█   ▀ ▓██▒    ▒██▒  ██▒ ██ ▀█   █    ▒████▄    ▓██▒
▓██    ▓██░▒███   ▒██░    ▒██░  ██▒▓██  ▀█ ██▒   ▒██  ▀█▄  ▒██▒
▒██    ▒██ ▒▓█  ▄ ▒██░    ▒██   ██░▓██▒  ▐▌██▒   ░██▄▄▄▄██ ░██░
▒██▒   ░██▒░▒████▒░██████▒░ ████▓▒░▒██░   ▓██░    ▓█   ▓██▒░██░
░ ▒░   ░  ░░░ ▒░ ░░ ▒░▓  ░░ ▒░▒░▒░ ░ ▒░   ▒ ▒     ▒▒   ▓▒█░░▓
░  ░      ░ ░ ░  ░░ ░ ▒  ░  ░ ▒ ▒░ ░ ░░   ░ ▒░     ▒   ▒▒ ░ ▒ ░
░      ░      ░     ░ ░   ░ ░ ░ ▒     ░   ░ ░      ░   ▒    ▒ ░
       ░      ░  ░    ░  ░    ░ ░           ░          ░  ░ ░

# Melon AI - Health & Wellness Assistant

A sophisticated AI-powered health and wellness platform that combines chatbot capabilities with meal analysis and personalized health recommendations.

## Project Overview

Melon AI is a full-stack application consisting of:
- Backend: FastAPI-based REST API with AI/ML capabilities
- Frontend: Modern React-based UI
- AI Features: Chatbot, meal analysis, and personalized health recommendations

## Tech Stack

### Backend
- Python 3.12
- FastAPI
- Chainlit
- LangChain
- Google Cloud Platform services
- Various ML/AI libraries

### Frontend
- React
- TypeScript
- Vite
- ESLint

## Setup & Installation

### Prerequisites
- Python 3.12+
- Node.js 16+
- Docker (optional)
- Google Cloud SDK (for deployment)

### Backend Setup

#bash


git clone [repository-url]
Navigate to backend directory
cd backend
Create and activate virtual environment
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Set up environment variables
cp .env.example .env
Edit .env with your configuration
Run the backend
python app.py
Setup


### Frontend Setup
bash
Navigate to frontend directory
cd ui
Install dependencies
npm install
Run development server
npm run dev


### Docker Setup
bash
Build and run using Docker
docker build -t melon-ai .
docker run -p 8080:8080 melon-ai


## API Endpoints

- `POST /chatbot/`: Main chatbot interaction endpoint
- `POST /register/`: User registration endpoint
- Health check: `GET /health`

## Features

1. **AI Chatbot**
   - Natural language processing
   - Context-aware responses
   - Health and wellness focused conversations

2. **Meal Analysis**
   - Image-based food recognition
   - Nutritional information
   - Dietary recommendations

3. **Health Monitoring**
   - Personal health tracking
   - Progress monitoring
   - Customized wellness plans

## Development

### Running Tests

bash
Backend tests
cd backend
pytest
Frontend tests
cd ui
npm test



### Code Style
- Backend: Black formatter, isort for imports
- Frontend: ESLint + Prettier

## Deployment

The application is configured for Google Cloud Platform deployment:

bash
Deploy to GCP
gcloud builds submit --config cloudbuild.yaml


## Environment Variables

Key environment variables needed:

LANGCHAIN_API_KEY=your_key
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
HF_TOKEN=your_token



## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Contact

Project Link: [https://github.com/thisisberu/melonai]


his README provides a comprehensive overview of your project while maintaining professionalism and including all necessary information for developers to get started. The ASCII art at the top adds a nice touch while keeping it professional.

````
