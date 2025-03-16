import streamlit as st
from retriever.script import hybrid_search
import time


st.set_page_config(
    page_title="News Retrieval System",
    page_icon="📰",
    layout="wide"
)

def create_app():
    
    st.title("📰 Hybrid News Retrieval System")
    
    st.markdown("""
    ### About This App
    
    This application uses advanced retrieval techniques to find the most relevant news articles based on your query:
    
    - **Hybrid Search**: Combines dense and sparse vector search for better results
    - **Dense Embeddings**: Generated using VoyageAI's multilingual model
    - **Sparse Embeddings**: Uses SPLADE for lexical search capabilities
    - **Binary Quantization**: Optimizes vector storage and improves search performance
    - **Rank Fusion**: Merges results using RRF (Reciprocal Rank Fusion)
    
    Simply enter your search query below to find relevant news articles.
    """)
    
    
    st.header("Search News Articles")
    query = st.text_input("Enter your search query:")
    top_k = st.slider("Number of results to display", min_value=1, max_value=20, value=5)
    
    if st.button("Search") or query:
        if query:
            with st.spinner("Searching articles..."):
                try:
                    
                    start_time = time.time()
                    
                    results = hybrid_search(query, top_k=top_k)
                    
                    elapsed_time = time.time() - start_time
                    
                    if not results:
                        st.warning("No results found for your query.")
                    else:
                        st.success(f"Found {len(results)} relevant articles in {elapsed_time:.4f} seconds")
                      
                        for i, point in enumerate(results, start=1):
                            with st.expander(f"Result #{i}: {point.payload.get('headline', 'Article')} (Score: {point.score:.4f})"):
                                
                                st.markdown(f"**Headline:** {point.payload.get('headline', 'N/A')}")
                                st.markdown(f"**Description:** {point.payload.get('short_description', 'N/A')}")
                               
                                link = point.payload.get('link', 'N/A')
                                st.markdown(f"**Link:** [{link}]({link})")
                                
                                category = point.payload.get('category', 'N/A')
                                st.markdown(f"**Category:** {category}")
                                
                                st.progress(min(point.score, 1.0))
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.info("Please enter a search query")
    
   
    st.markdown("---")
    st.markdown("Built with Streamlit, Qdrant, and VoyageAI")

if __name__ == "__main__":
    create_app()
