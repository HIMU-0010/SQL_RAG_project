from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from few_shots import few_shots

import os
import logging
from typing import Optional
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class DatabaseConnectionError(Exception):
    """Custom exception for database connection issues"""
    pass


class LLMInitializationError(Exception):
    """Custom exception for LLM initialization issues"""
    pass


def create_database_connection():
    """Create database connection with error handling"""
    try:
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD") 
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        
        connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
        
        db = SQLDatabase.from_uri(
            connection_string,
            sample_rows_in_table_info=3,
            include_tables=None,
        )
        
        # Test the connection
        db.get_usable_table_names()
        logger.info("Database connection established successfully")
        return db
        
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise DatabaseConnectionError(f"Database connection failed: {str(e)}")


def initialize_llm():
    """Initialize LLM with error handling"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMInitializationError("GOOGLE_API_KEY not found in environment variables")
        
        
        llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=2048,
        )
        
        logger.info("LLM initialized successfully")
        return llm
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise LLMInitializationError(f"LLM initialization failed: {str(e)}")


def create_example_selector():
    """Create semantic similarity example selector with error handling"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        
        to_vectorize = [" ".join(example.values()) for example in few_shots]
        
        vectorstore = Chroma.from_texts(
            texts=to_vectorize,
            embedding=embeddings,
            metadatas=few_shots,
            persist_directory=None
        )
        
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=2,
        )
        
        logger.info("Example selector created successfully")
        return example_selector
        
    except Exception as e:
        logger.error(f"Failed to create example selector: {str(e)}")
        raise Exception(f"Example selector creation failed: {str(e)}")


def create_custom_prompt():
    """Create custom prompt template for few-shot learning"""
    
    mysql_prompt = """You are a MySQL expert. Given an input question, create a syntactically correct MySQL query.

    Instructions:
    - Query for at most {top_k} results using LIMIT clause
    - Only query columns needed to answer the question
    - Wrap column names in backticks (`)
    - Use only existing column names from the tables
    - Use CURDATE() for current date queries
    - Return the query and explain your reasoning

    Database Schema: {table_info}

    Here are some examples of good queries:
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    try:
        example_selector = create_example_selector()
        
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=mysql_prompt,
            suffix="\nQuestion: {input}",
            input_variables=["input", "table_info", "top_k"],
        )
        
        return few_shot_prompt
        
    except Exception as e:
        logger.error(f"Failed to create prompt: {str(e)}")
        # Return a simple prompt as fallback
        return PromptTemplate(
            input_variables=["input", "table_info"],
            template="Given the database schema: {table_info}\n\nAnswer this question: {input}"
        )


def get_sql_agent():
    """Create SQL agent using SQLDatabaseToolkit with comprehensive error handling"""
    try:
        db = create_database_connection()
        llm = initialize_llm()
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        tools = toolkit.get_tools()
        
        system_message = """You are a SQL expert assistant. Use the available tools to:

        1. First, examine the database schema to understand table structures
        2. Create appropriate SQL queries based on the user's question
        3. Execute the queries and interpret results
        4. Provide clear, accurate answers

        Always:
        - Use backticks around column names
        - Limit results appropriately 
        - Double-check table and column names exist
        - Explain your reasoning

        Available tools: sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker
        """
        
        # Create SQL agent
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            max_execution_time=30,
            early_stopping_method="generate",
            handle_parsing_errors=True,
        )
        
        logger.info("SQL agent created successfully")
        return agent
        
    except DatabaseConnectionError as e:
        logger.error(f"Database error: {str(e)}")
        raise
    except LLMInitializationError as e:
        logger.error(f"LLM error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating SQL agent: {str(e)}")
        raise Exception(f"Failed to create SQL agent: {str(e)}")


def query_database(question: str, agent=None) -> dict:
    """Query database with error handling and response formatting"""
    try:
        if agent is None:
            agent = get_sql_agent()
        
        logger.info(f"Processing question: {question}")
        
        # Execute query through agent with intermediate steps
        response = agent.invoke(
            {"input": question}
        )
        
        # Extract intermediate steps from the response
        intermediate_steps = []
        
        # Check if the response has intermediate_steps attribute
        if hasattr(response, 'intermediate_steps'):
            intermediate_steps = response.intermediate_steps
        elif isinstance(response, dict) and 'intermediate_steps' in response:
            intermediate_steps = response['intermediate_steps']
        
        # Format thought process
        thought_process = []
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) >= 2:
                action, observation = step[0], step[1]
                
                # Extract tool information
                tool_name = 'unknown'
                tool_input = ''
                
                if hasattr(action, 'tool'):
                    tool_name = action.tool
                if hasattr(action, 'tool_input'):
                    tool_input = str(action.tool_input)
                
                thought_process.append({
                    "action": str(action),
                    "tool": tool_name,
                    "tool_input": tool_input,
                    "observation": str(observation)
                })
        
        # If no intermediate steps found, try alternative approach
        if not thought_process and hasattr(agent, 'agent') and hasattr(agent.agent, 'llm_chain'):
            # This is a fallback - you might need to enable verbose mode differently
            logger.warning("No intermediate steps captured - thought process will be empty")
        
        # Format response
        result = {
            "success": True,
            "question": question,
            "answer": response.get("output", str(response)) if isinstance(response, dict) else str(response),
            "thought_process": thought_process,
            "error": None
        }
        
        logger.info("Query processed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        return {
            "success": False,
            "question": question,
            "answer": None,
            "thought_process": [],
            "error": str(e)
        }

# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the system
        agent = get_sql_agent()
        
        # Example query
        test_question = "How many t-shirts are in stock?"
        result = query_database(test_question, agent)
        
        if result["success"]:
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"System initialization failed: {str(e)}")
        print("\nPlease check:")
        print("1. Environment variables are set correctly")
        print("2. Database is accessible")
        print("3. Google API key is valid")