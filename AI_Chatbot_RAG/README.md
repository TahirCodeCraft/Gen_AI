# Personal Chatbot

## Overview

The Personal Chatbot is a Streamlit-based application that allows users to upload multiple documents (PDF, CSV, DOCX, TXT) and interact with a chatbot to ask questions based on the content of these documents. The chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate and context-aware responses.

## Features

- **Multi-Document Support**: Upload and query multiple documents of different formats.
- **Context-Aware Responses**: Maintains the context of each document to provide accurate answers.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and easy-to-use interface.
- **Customizable**: Easily extendable to support additional document types and features.

## Requirements

- Python 3.8 or higher
- Streamlit
- LangChain
- HuggingFace Transformers
- docx2txt
- sentence-transformers
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/TahirCodeCraft/Gen_AI.git
   cd your-repository-name
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory of the project.
   - Add your API keys and other environment variables to the `.env` file. Example:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

## Usage

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Documents**:
   - Use the file uploader to upload PDF, CSV, DOCX, or TXT files.
   - The chatbot will process the documents and create a vector store for querying.

3. **Ask Questions**:
   - Enter your questions in the chat interface.
   - The chatbot will provide responses based on the content of the uploaded documents.

## Code Structure

- **`Ai_chatbot.py`**: Main application file that sets up the Streamlit interface and handles user interactions.
- **`requirements.txt`**: List of Python dependencies required for the project.
- **`..\.env`**: Environment variables for configuration.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with a descriptive message.
4. Push your branch to your fork.
5. Open a pull request describing your changes.

## Acknowledgements

- Special thanks to the developers of Streamlit, LangChain, and HuggingFace for their excellent tools and libraries.