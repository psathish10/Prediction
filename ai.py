import streamlit as st
from kb import SalesKnowledgeBase

def sales_data_chat_interface():
    """
    Streamlit interface for interacting with sales knowledge base
    """
    st.title("Schwing Stetter Sales Data Intelligence")
    
    # Initialize knowledge base
    kb = SalesKnowledgeBase()
    
    # Chat input
    user_query = st.text_input("Ask a question about sales data:")
    
    if user_query:
        with st.spinner('Analyzing sales data...'):
            # Query knowledge base
            response = kb.query_knowledge_base(user_query)
            
            # Display response
            st.markdown("### Insights")
            st.info(response)
    
    # Sidebar for additional interactions
    st.sidebar.title("Data Insights")
    
    if st.sidebar.button("Generate Data Summary"):
        summary = kb.generate_data_summary()
        st.sidebar.markdown("### Data Summary")
        st.sidebar.write(summary)

def main():
    sales_data_chat_interface()

if __name__ == "__main__":
    main()