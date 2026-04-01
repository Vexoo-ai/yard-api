import os
import anthropic
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Anthropic client
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


async def call_claude_llm_stream(query, search_results):
    """Stream Claude responses for web search results"""
    try:
        formatted_results = format_web_search_results(search_results)

        context_parts = []
        for r in formatted_results:
            source = r.get('source') or 'Unknown Source'
            title = r.get('title') or 'No Title'
            snippet = r.get('snippet') or 'No Snippet Available'
            link = r.get('link', 'N/A')
            context_parts.append(
                f"Source: {source}\nTitle: {title}\nSnippet: {snippet}\nURL: {link}"
            )

        context = '\n\n'.join(
            context_parts) if context_parts else "No search results available."

        system_message = """You are an advanced AI assistant with real-time internet search capabilities,
designed to provide exceptionally detailed, comprehensive, and insightful responses to any type of query.
Your knowledge spans a wide range of topics including but not limited to general knowledge, current events,
science, technology, programming, arts, and more. Your responses should be characterized by their depth,
precision, and attention to nuanced details. Always cite the sources provided when referencing information."""

        user_content = f"Internet Search Results:\n{context}\n\nCurrent Question: {query}"

        logger.info(f"Creating Claude stream for query: {query}")

        with claude_client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=system_message,
            messages=[{"role": "user", "content": user_content}]
        ) as stream:
            for text in stream.text_stream:
                yield text

    except Exception as e:
        logger.error(f"Error in Claude LLM stream: {str(e)}", exc_info=True)
        yield f"\nError calling Claude AI: {str(e)}\n"
        fallback = await fallback_response(query, search_results)
        yield fallback


async def fallback_response(query, search_results):
    """Generate a fallback response using search results when Claude fails"""
    formatted_results = format_web_search_results(search_results)
    response = (
        f"I apologize, but I couldn't access the AI language model due to an error. "
        f"However, here is a summary of the top search results for your query: '{query}'\n\n"
    )
    for idx, result in enumerate(formatted_results[:5], 1):
        response += (
            f"{idx}. {result['title']}\n"
            f"   Source: {result['source']}\n"
            f"   Snippet: {result['snippet'][:150]}...\n\n"
        )
    response += "For more detailed information, please visit the source websites or try your query again later."
    return response


def format_web_search_results(search_data):
    """Format search results for use in prompts"""
    formatted_results = []

    if not search_data or not isinstance(search_data, dict):
        return []

    organic_results = search_data.get('organic_results', []) or []

    for result in organic_results:
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
            'date': result.get('date'),
            'title': result.get('title', 'No Title'),
            'snippet': result.get('snippet', 'No Snippet Available'),
            'highlight': result.get('snippet_highlighted_words', ''),
            'engine': result.get('engine', 'Unknown'),
            'link': result.get('link', 'No URL')
        }
        formatted_results.append(formatted_result)

    return formatted_results
