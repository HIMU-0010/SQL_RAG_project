# 🏪 AtliQ T-Shirts Database Q&A 👕

A Streamlit-based application that allows users to query a t-shirt inventory database using natural language. The system leverages AI models (IBM Watson and Google Gemini) to convert natural language questions into SQL queries and provide intelligent responses.

## ✨ Features

- **Natural Language Queries**: Ask questions about t-shirt inventory in plain English
- **AI-Powered SQL Generation**: Converts natural language to SQL using IBM Watson or Google Gemini
- **Few-Shot Learning**: Uses semantic similarity to find relevant examples for better query generation
- **Vector Database**: ChromaDB for storing and retrieving similar query examples
- **Interactive Web Interface**: Clean Streamlit UI with example questions and error handling
- **Database Management**: Complete MySQL database setup for t-shirt inventory

## 📁 Project Structure

```
GENAI-SQL/
├── .env                                    # Environment variables (API keys, DB config)
├── .env.example                           # Template for environment variables
├── .gitignore                            # Git ignore file
├── chroma_db/                            # ChromaDB vector database directory
│   ├── e8ff5cf5-6bac-4393-...          # ChromaDB data files
│   └── chroma.sqlite3                   # ChromaDB SQLite database
├── database/
│   └── db_creation_atliq_t_shirts.sql  # Database schema and sample data
├── few_shots.py                         # Few-shot learning examples
├── langchain_helper_Google_gemini.py   # Google Gemini LLM implementation
├── langchain_helper_IBM_watson.py      # IBM Watson LLM implementation (active)
├── main.py                              # Main Streamlit application
└── requirements.txt                     # Python dependencies
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SQL_RAG_project
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup
```bash
# Start MySQL server and create the database
mysql -u root -p < database/db_creation_atliq_t_shirts.sql
```

### 5. Environment Configuration
Create a `.env` file in the root directory:
```env
# IBM Watsonx AI API Key (Primary)
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://au-syd.ml.cloud.ibm.com

# Google AI API Key (Optional)
GOOGLE_API_KEY=your_google_api_key_here

# Database Configuration
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_NAME=atliq_tshirts
```

### 6. Initialize Vector Database
```bash
# Run preprocessing to set up ChromaDB
python preprocess.py
```

### 7. Launch Application
```bash
# Start the Streamlit application
streamlit run main.py
```

## ⚙️ Configuration

### API Keys Required:
- **IBM Watson**: Sign up at [IBM Cloud](https://cloud.ibm.com/) and create a Watson Machine Learning service
- **Google Gemini** (Optional): Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Database Configuration:
- Ensure MySQL is running on localhost:3306
- Database name: `atliq_tshirts`
- Default user: `root` (modify in `.env` as needed)

## 🎯 Usage

1. **Start the Application**: Run `streamlit run main.py`
2. **Ask Questions**: Type natural language questions about the t-shirt inventory
3. **View Results**: Get AI-generated answers with SQL query explanations
4. **Use Examples**: Click on sidebar examples for quick queries

## 💡 Example Questions

- "How many t-shirts are in stock?"
- "What are the different sizes available?"
- "Show me all white t-shirts"
- "What's the total inventory value?"
- "How many Levi's t-shirts do we have?"
- "What sizes are available in Nike?"
- "Show me t-shirts under $50"
- "How much revenue will we generate from all Adidas t-shirts with discounts?"

## 🤖 AI Models

### Current: IBM Watson (Granite-8B-Code-Instruct)
- **Model**: `ibm/granite-8b-code-instruct`
- **Temperature**: 0.1 (for consistent SQL generation)
- **Decoding**: Greedy method

### Alternative: Google Gemini 1.5 Flash
- **Model**: `gemini-1.5-flash`
- **Temperature**: 0.1
- **Max Tokens**: 2048

## 🗄️ Database Schema

### Tables:
1. **t_shirts**
   - `t_shirt_id` (Primary Key)
   - `brand` (Van Huesen, Levi, Nike, Adidas)
   - `color` (Red, Blue, Black, White)
   - `size` (XS, S, M, L, XL)
   - `price` (10-50)
   - `stock_quantity`

2. **discounts**
   - `discount_id` (Primary Key)
   - `t_shirt_id` (Foreign Key)
   - `pct_discount` (0-100%)


**Built with**: Python, Streamlit, LangChain, IBM Watson, Google Gemini, ChromaDB, MySQL

## Happy Coding 🚀