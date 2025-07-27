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
            model_id="ibm/granite-8b-code-instruct",
            url=watsonx_url,
            apikey=watsonx_api_key,
            project_id=watsonx_project_id,
            params={
                "decoding_method": "greedy",
                "temperature": 0.1,
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
    """Create SQL agent with WatsonX - simplified and fixed version"""
    try:
        db = get_database()
        llm = get_watsonx_llm()
        
        # Create the agent with minimal, working configuration
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,  # Reduced iterations
            # Remove unsupported early_stopping_method
            return_intermediate_steps=True
        )
        
        logger.info("SQL agent created successfully")
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
    """Execute query using the SQL agent with improved error handling"""
    try:
        logger.info(f"Processing question: {question}")
        
        # Try the agent first
        try:
            result = agent.invoke({"input": question})
            return {
                "success": True,
                "answer": result.get("output", "No answer generated"),
                "intermediate_steps": result.get("intermediate_steps", [])
            }
        except Exception as agent_error:
            logger.warning(f"Agent failed: {str(agent_error)}")
            # Fall back to direct SQL approach
            return query_with_fallback(question)
        
    except Exception as e:
        logger.error(f"Error in query_database: {str(e)}")
        return query_with_fallback(question)

def query_with_fallback(question):
    """Query using direct SQL generation - more reliable fallback"""
    try:
        logger.info("Using fallback query method")
        db = get_database()
        llm = get_watsonx_llm()
        
        # Get relevant examples for context
        try:
            relevant_examples = get_relevant_examples(question, k=1)
            example_context = ""
            if relevant_examples:
                example = relevant_examples[0]
                example_context = f"""
Here's a similar example:
Question: {example['Question']}
SQL Query: {example['SQLQuery']}
Answer: {example['Answer']}
"""
        except:
            example_context = ""
        
        # Direct prompt for SQL generation
        sql_prompt = f"""You are a MySQL expert. Generate a SQL query to answer this question about a t-shirt database.

Database Schema:
- t_shirts table: t_shirt_id, brand, color, size, price, stock_quantity
- discounts table: t_shirt_id, pct_discount

Available values:
- Brands: Van Huesen, Levi, Nike, Adidas  
- Colors: Red, Blue, Black, White
- Sizes: XS, S, M, L, XL

{example_context}

Question: {question}

Generate only the SQL query, no explanations:"""

        # Get SQL query from LLM
        sql_response = llm.invoke(sql_prompt)
        sql_query = sql_response.strip()
        
        # Clean up the SQL query
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].strip()
        
        # Remove any extra text before SELECT
        if "SELECT" in sql_query.upper():
            sql_query = "SELECT" + sql_query.upper().split("SELECT", 1)[1]
            sql_query = sql_query.replace("SELECT", "SELECT", 1)  # Keep original case for first SELECT
        
        logger.info(f"Generated SQL: {sql_query}")
        
        # Execute the query
        result = db.run(sql_query)
        logger.info(f"SQL Result: {result}")
        
        # Generate natural language answer
        answer_prompt = f"""Based on this SQL query result, provide a clear, natural language answer to the user's question.

Question: {question}
SQL Query: {sql_query}
SQL Result: {result}

Provide a concise, helpful answer:"""
        
        answer_response = llm.invoke(answer_prompt)
        
        return {
            "success": True,
            "answer": answer_response.strip(),
            "sql_query": sql_query,
            "sql_result": str(result)
        }
        
    except Exception as e:
        logger.error(f"Fallback query failed: {str(e)}")
        return {
            "success": False,
            "error": f"Unable to process question: {str(e)}. Please try rephrasing your question.",
            "answer": None
        }

def get_fallback_chain():
    """Create a simpler database chain as fallback"""
    try:
        db = get_database()
        llm = get_watsonx_llm()
        
        # Create a simple prompt template with correct input variable
        template = """Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer.
        Use the following format:

        Question: {query}
        SQLQuery: [Your SQL query here]
        SQLResult: [Result of the SQL query]
        Answer: [Final answer based on the result]

        Only use the following tables:
        {table_info}

        Question: {query}"""
        
        prompt = PromptTemplate(
            input_variables=["query", "table_info"],  # Fixed: use "query" instead of "question"
            template=template
        )
        
        chain = SQLDatabaseChain.from_llm(
            llm=llm,
            db=db,
            prompt=prompt,
            verbose=True,
            return_intermediate_steps=True,
            use_query_checker=True,
            return_sql=True
        )
        
        logger.info("Fallback database chain created successfully")
        return chain
    except Exception as e:
        logger.error(f"Failed to create fallback chain: {str(e)}")
        raise

def query_with_chain_fallback(question):
    """Alternative fallback using the chain method"""
    try:
        logger.info("Using chain fallback method")
        chain = get_fallback_chain()
        result = chain.invoke({"query": question})  # Use "query" key to match the prompt template
        
        return {
            "success": True,
            "answer": result.get("result", "No answer generated"),
            "sql_query": result.get("intermediate_steps", [{}])[-1].get("sql_cmd", "") if result.get("intermediate_steps") else "",
            "sql_result": result.get("intermediate_steps", [{}])[-1].get("result", "") if result.get("intermediate_steps") else ""
        }
    except Exception as e:
        logger.error(f"Chain fallback query failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "answer": f"Unable to process question: {str(e)}. Please try rephrasing your question."
        }