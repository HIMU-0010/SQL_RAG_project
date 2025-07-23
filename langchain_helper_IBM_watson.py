import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from few_shots import few_shots
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_watsonx_llm():
    """Initialize and return WatsonX LLM"""
    try:
        # Get environment variables with validation
        watsonx_url = os.getenv("WATSONX_URL")
        watsonx_api_key = os.getenv("WATSONX_API_KEY")
        watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
        
        # Validate required environment variables
        if not watsonx_url:
            raise ValueError("WATSONX_URL environment variable is required")
        if not watsonx_api_key:
            raise ValueError("WATSONX_API_KEY environment variable is required")
        if not watsonx_project_id:
            raise ValueError("WATSONX_PROJECT_ID environment variable is required")
        
        # Watsonx configuration
        watsonx_llm = WatsonxLLM(
            model_id="meta-llama/llama-2-70b-chat",  # or your preferred model
            url=watsonx_url,
            apikey=watsonx_api_key,
            project_id=watsonx_project_id,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 500,
                "temperature": 0.1,
                "stop_sequences": ["\n\n"]
            }
        )
        logger.info("WatsonX LLM initialized successfully")
        return watsonx_llm
    except Exception as e:
        logger.error(f"Failed to initialize WatsonX LLM: {str(e)}")
        raise

def get_database():
    """Create database connection"""
    try:
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        
        db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
        logger.info("Database connection established")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

def create_vector_db():
    """Create Chroma vector database with few-shot examples"""
    try:
        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store with few-shot examples
        to_vectorize = [
            " ".join([example["Question"], example["SQLQuery"]]) 
            for example in few_shots
        ]
        
        vectorstore = Chroma.from_texts(
            texts=to_vectorize,
            embedding=embeddings,
            metadatas=few_shots,
            persist_directory="./chroma_db"
        )
        
        logger.info("Vector database created successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to create vector database: {str(e)}")
        raise

def get_few_shot_db_chain():
    """Create few-shot database chain with WatsonX and Chroma vector DB"""
    try:
        db = get_database()
        llm = get_watsonx_llm()
        
        # Create vector database
        vectorstore = create_vector_db()
        
        # Create embeddings for example selection
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create semantic similarity example selector
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=2,  # Select top 2 most similar examples
        )
        
        # Define the example prompt
        example_prompt = PromptTemplate(
            input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
            template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
        )
        
        # Create few-shot prompt template
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="""You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run.
            Unless the user specifies in their question a specific number of examples they wish to obtain, 
            always limit your query to at most 10 results using the LIMIT clause.
            
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            Pay attention to use only the column names that you can see in the schema description.
            Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            
            Use the following format:
            Question: Question here
            SQLQuery: SQL Query to run
            SQLResult: Result of the SQL query  
            Answer: Final answer here
            
            Only use the following tables:
            {table_info}
            
            Below are a number of examples of questions and their corresponding SQL queries.""",
            suffix="\nQuestion: {input}",
            input_variables=["input", "table_info"],
        )
        
        chain = SQLDatabaseChain.from_llm(
            llm=llm,
            db=db,
            prompt=few_shot_prompt,
            verbose=True,
            return_intermediate_steps=True
        )
        
        logger.info("Few-shot database chain with vector DB created successfully")
        return chain
    except Exception as e:
        logger.error(f"Failed to create database chain: {str(e)}")
        raise

def get_sql_agent():
    """Create SQL agent with WatsonX and vector-enhanced prompting"""
    try:
        db = get_database()
        llm = get_watsonx_llm()
        
        # Create vector database for context retrieval
        vectorstore = create_vector_db()
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        # Create custom system message with vector-enhanced context
        system_message = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct MySQL query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        
        Database Schema Context:
        - t_shirts table: t_shirt_id, brand, color, size, price, stock_quantity
        - discounts table: t_shirt_id, pct_discount
        
        When calculating revenue with discounts, use proper JOIN operations.
        Always provide numerical answers for quantities and prices."""
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="zero-shot-react-description",
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
        
        logger.info("SQL agent with vector enhancement created successfully")
        return agent_executor
    except Exception as e:
        logger.error(f"Failed to create SQL agent: {str(e)}")
        raise

def get_relevant_examples(question, k=2):
    """Retrieve relevant examples from vector database"""
    try:
        vectorstore = create_vector_db()
        relevant_docs = vectorstore.similarity_search(question, k=k)
        
        relevant_examples = []
        for doc in relevant_docs:
            relevant_examples.append(doc.metadata)
        
        return relevant_examples
    except Exception as e:
        logger.error(f"Failed to retrieve relevant examples: {str(e)}")
        return []

def query_database(question, agent):
    """Execute query using the SQL agent with vector-enhanced context"""
    try:
        logger.info(f"Processing question: {question}")
        
        # Get relevant examples from vector database
        relevant_examples = get_relevant_examples(question, k=2)
        
        # Build context with relevant examples
        examples_context = ""
        if relevant_examples:
            examples_context = "\n\nHere are some similar examples:\n"
            for i, example in enumerate(relevant_examples, 1):
                examples_context += f"\nExample {i}:\n"
                examples_context += f"Question: {example['Question']}\n"
                examples_context += f"SQL: {example['SQLQuery']}\n"
                examples_context += f"Answer: {example['Answer']}\n"
        
        # Enhanced context prompt with vector-retrieved examples
        context_prompt = f"""
        You are working with an AtliQ T-shirts database. The database contains information about t-shirt inventory including:
        - t_shirts table: t_shirt_id, brand, color, size, price, stock_quantity
        - discounts table: t_shirt_id, pct_discount
        
        {examples_context}
        
        Please answer this question about the database: {question}
        
        If you need to calculate revenue with discounts, join the tables appropriately.
        Always provide clear, numerical answers when dealing with quantities or prices.
        """
        
        result = agent.invoke({"input": context_prompt})
        
        return {
            "success": True,
            "answer": result.get("output", "No answer generated"),
            "intermediate_steps": result.get("intermediate_steps", []),
            "relevant_examples": relevant_examples
        }
    except Exception as e:
        logger.error(f"Error querying database: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "answer": None
        }