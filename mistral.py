from mistralai_azure import MistralAzure
import os
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MistralAzure client
mistral_client = MistralAzure(
    azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
    azure_api_key=os.getenv("AZURE_AI_API_KEY")
)

async def call_mistral_llm_stream(query, search_results):
    """Stream Mistral LLM responses using the client-based approach"""
    try:
        # Format search results
        formatted_results = format_web_search_results(search_results)
        
        # Build context string
        context_parts = []
        for r in formatted_results:
            source = r.get('source') or 'Unknown Source'
            title = r.get('title') or 'No Title'
            snippet = r.get('snippet') or 'No Snippet Available'
            link = r.get('link', 'N/A')
            
            context_parts.append(f"Source: {source}\nTitle: {title}\nSnippet: {snippet}\nURL: {link}")
        
        context = '\n\n'.join(context_parts) if context_parts else "No search results available."
        
        # System message
        system_message = """You are an advanced AI assistant with real-time internet search capabilities, designed to provide exceptionally detailed, comprehensive, and insightful responses to any type of query. Your knowledge spans a wide range of topics including but not limited to general knowledge, current events, science, technology, programming, arts, and more. Your responses should be characterized by their depth, precision, and attention to nuanced details."""
        
        # Format messages for Mistral
        mistral_messages = [{"role": "system", "content": system_message}]
        
        # Add context and query as user message
        user_content = f"Internet Search Results:\n{context}\n\nCurrent Question: {query}"
        mistral_messages.append({"role": "user", "content": user_content})
        
        # Create stream using the correct method
        logger.info(f"Creating Mistral stream for query: {query}")
        
        # Since the Mistral client is synchronous, run it in a separate thread
        # to avoid blocking the async event loop
        stream = mistral_client.chat.stream(
            model="azureai",
            messages=mistral_messages,
            max_tokens=8000,
            temperature=0.7
        )
        
        # Process the synchronous stream in a way that's compatible with async
        for chunk in stream:
            try:
                # Extract content from the stream
                if hasattr(chunk, 'data') and chunk.data:
                    completion_chunk = chunk.data
                    if hasattr(completion_chunk, 'choices') and completion_chunk.choices:
                        delta = completion_chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content is not None:
                            yield delta.content
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in Mistral LLM stream: {str(e)}", exc_info=True)
        yield f"\nError calling Mistral AI: {str(e)}\n"
        fallback = await fallback_response(query, search_results)
        yield fallback

async def fallback_response(query, search_results):
    """Generate a fallback response using search results"""
    formatted_results = format_web_search_results(search_results)
    response = f"I apologize, but I couldn't access the AI language model due to an error. However, I can provide you with a summary of the top search results for your query: '{query}'\n\n"
    
    for idx, result in enumerate(formatted_results[:5], 1):
        response += f"{idx}. {result['title']}\n   Source: {result['source']}\n   Snippet: {result['snippet'][:150]}...\n\n"
    
    response += "For more detailed information, please visit the source websites or try your query again later when the AI model is available."
    return response

def format_web_search_results(search_data):
    """Format search results for use in prompts"""
    formatted_results = []
    
    # Ensure search_data is properly structured
    if not search_data or not isinstance(search_data, dict):
        return []
    
    organic_results = search_data.get('organic_results', []) or []
    
    for result in organic_results:
        # Handle potentially missing fields
        if not isinstance(result, dict):
            continue
            
        displayed_link = result.get('displayed_link', '')
        source = None
        if displayed_link:
            if '://' in displayed_link:
                displayed_link = displayed_link.split('://', 1)[-1]
            source = displayed_link.split('/')[0]

        formatted_result = {
            'source': source or 'Unknown Source',
            'date': result.get('date'),  # Can be None
            'title': result.get('title', 'No Title'),
            'snippet': result.get('snippet', 'No Snippet Available'),
            'highlight': result.get('snippet_highlighted_words', ''),
            'engine': result.get('engine', 'Unknown'),
            'link': result.get('link', 'No URL')
        }
        formatted_results.append(formatted_result)
    
    return formatted_results