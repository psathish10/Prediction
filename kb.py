import pandas as pd
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

class SalesKnowledgeBase:
    def __init__(self, csv_path='schwing_stetter_sales.csv'):
        """
        Initialize Knowledge Base for Sales Data
        
        Args:
            csv_path (str): Path to the sales CSV file
        """
        # Load environment variables
        load_dotenv()
        
        # Configure API key
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in .env file.")
        
        genai.configure(api_key=self.api_key)
        
        # Class attributes
        self.csv_path = csv_path
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store_path = "sales_faiss_index"
        
        # Load data
        self.df = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load sales data from CSV
        
        Returns:
            pd.DataFrame: Loaded sales data
        """
        try:
            df = pd.read_csv(self.csv_path)
            return df
        except FileNotFoundError:
            print(f"CSV file not found at {self.csv_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    def generate_data_summary(self) -> str:
        """
        Generate a comprehensive summary of the sales data
        
        Returns:
            str: Detailed summary of sales data
        """
        if self.df.empty:
            return "No data available for summary."
        
        summary = f"""
        Schwing Stetter Sales Data Summary (2022-2024):
        
        1. Dataset Overview:
        - Total Records: {len(self.df)}
        - Date Range: {self.df['Date'].min()} to {self.df['Date'].max()}
        
        2. Financial Metrics:
        - Total Revenue: €{self.df['Total_Sale_EUR'].sum():,.2f}
        - Average Order Value: €{self.df['Total_Sale_EUR'].mean():,.2f}
        - Median Order Value: €{self.df['Total_Sale_EUR'].median():,.2f}
        
        3. Product Insights:
        - Unique Product Categories: {self.df['Product_Category'].nunique()}
        - Top 3 Product Categories by Revenue:
        {self.df.groupby('Product_Category')['Total_Sale_EUR'].sum().nlargest(3).to_string()}
        
        4. Geographical Distribution:
        - Unique Countries: {self.df['Country'].nunique()}
        - Top 3 Countries by Sales:
        {self.df.groupby('Country')['Total_Sale_EUR'].sum().nlargest(3).to_string()}
        
        5. Sales Characteristics:
        - Total Units Sold: {self.df['Quantity'].sum()}
        - Average Quantity per Order: {self.df['Quantity'].mean():.2f}
        - Discount Analysis:
          * Average Discount: {self.df['Discount_Percentage'].mean():.2f}%
          * Max Discount: {self.df['Discount_Percentage'].max():.2f}%
        """
        return summary
    
    def process_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding
        
        Args:
            text (str): Input text to be split
        
        Returns:
            List[str]: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, 
            chunk_overlap=1000
        )
        return text_splitter.split_text(text)
    
    def create_vector_store(self) -> bool:
        """
        Create vector store from data summary
        
        Returns:
            bool: Whether vector store was created successfully
        """
        try:
            # Ensure the directory exists
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # Generate summary and process text
            summary = self.generate_data_summary()
            text_chunks = self.process_text(summary)
            
            # Create and save vector store
            vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
            vector_store.save_local(self.vector_store_path)
            
            print("Vector store created successfully!")
            return True
        
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False
    
    def query_knowledge_base(self, query: str) -> str:
        """
        Query the sales knowledge base
        
        Args:
            query (str): User's query about sales data
        
        Returns:
            str: Answer from the knowledge base
        """
        try:
            # Check if vector store exists, create if not
            if not os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
                print("Vector store not found. Creating now...")
                if not self.create_vector_store():
                    return "Could not create vector store. Please check your data and configuration."
            
            # Load existing vector store
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Perform similarity search
            similar_docs = vector_store.similarity_search(query, k=2)
            
            # Set up QA chain
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", 
                temperature=0.1,
                api_key=self.api_key
            )
            
            prompt_template = """
            Answer the sales data question as comprehensively as possible based on the context. 
            Provide detailed insights while ensuring accuracy.
            
            Context: {context}
            Question: {question}
            
            Detailed Answer:
            """
            
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            response = chain.run(input_documents=similar_docs, question=query)
            
            return response
        
        except Exception as e:
            return f"Error querying knowledge base: {e}"

def main():
    # Initialize knowledge base
    kb = SalesKnowledgeBase()
    
    # Ensure vector store is created
    kb.create_vector_store()
    
    # Example queries
    queries = [
        "What are the top-performing product categories?",
        "Which countries generate the most revenue?",
        "Provide insights on sales trends in 2023",
        "What is the average order value?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = kb.query_knowledge_base(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()