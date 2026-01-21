import os
import base64
import argparse
from dotenv import load_dotenv
import anthropic
from mistralai import Mistral

# Load environment variables
load_dotenv()

def process_document_with_mistral(client, file_path):
    """
    Process a document using Mistral's document understanding capabilities.
    
    Args:
        client: Mistral client
        file_path (str): Path to the document to process
        
    Returns:
        str: Extracted text content
    """
    try:
        print(f"Processing document with Mistral: {file_path}")
        
        # Upload the file
        with open(file_path, "rb") as file_content:
            uploaded_file = client.files.upload(
                file={
                    "file_name": os.path.basename(file_path),
                    "content": file_content,
                },
                purpose="ocr"
            )
        print(f"File uploaded successfully: {uploaded_file.id}")
        
        # Get the signed URL for the uploaded file
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
        print(f"Signed URL generated successfully")
        
        # Define the messages for document understanding
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract all text content from this document and format it as markdown. Include ALL text, preserve structure, and don't add any of your own commentary."
                    },
                    {
                        "type": "document_url",
                        "document_url": signed_url.url
                    }
                ]
            }
        ]
        
        # Get the chat response using document understanding
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages
        )
        
        # Extract the content from the response
        content = chat_response.choices[0].message.content
        print("Document processed successfully with Mistral")
        return content
        
    except Exception as e:
        print(f"Error processing document with Mistral: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def read_markdown_file(file_path):
    """
    Read markdown content from a file.
    
    Args:
        file_path (str): Path to the markdown file
        
    Returns:
        str: Markdown content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading markdown file: {e}")
        return None

def save_markdown(content, output_path):
    """
    Save content as markdown.
    
    Args:
        content: Content to save
        output_path (str): Path to save the markdown file
    """
    try:
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Markdown saved to {output_path}")
        
        # Print a short preview of the content
        preview_length = min(500, len(content))
        print(f"\nContent preview (first {preview_length} chars):")
        print(content[:preview_length] + "..." if len(content) > preview_length else content)
        
    except Exception as e:
        print(f"Error saving markdown: {e}")
        import traceback
        print(traceback.format_exc())

def stream_answer_with_claude(client, markdown_content, query):
    """
    Stream the answer from Claude with extended thinking mode, properly handling the event types.
    
    Args:
        client: Anthropic Claude client
        markdown_content (str): Extracted text content from the document as markdown
        query (str): Question to ask about the document
    """
    try:
        print(f"Answering question with Claude (extended thinking mode): {query}")
        
        # Define a more comprehensive system prompt that encourages thorough scanning
        system_message = """
        You are an expert document analyst with exceptional attention to detail. Your task is to answer questions about the provided document by performing a comprehensive analysis of ALL content.
        
        Follow these guidelines:
        1. Scan the ENTIRE document thoroughly, page by page, examining all sections, tables, lists, and paragraphs
        2. Look for ALL instances of information relevant to the query, not just the first occurrence
        3. Be meticulous in your search - make sure to capture every relevant detail
        4. Organize your findings in a clear, structured format
        5. If information seems to be listed in multiple places or formats, consolidate it in your answer
        6. Indicate clearly which pages or sections the information was found in
        7. If you cannot find information related to the query, explicitly state that after a thorough examination
        8. Base your answers solely on information in the document, not on external knowledge
        
        When asked to list or extract information, be particularly comprehensive and ensure you've examined every part of the document.
        """
        
        # Create a more explicit prompt for more thorough analysis
        prompt = f"""
        I need you to analyze this document extremely carefully and answer the following question:
        
        "{query}"
        
        Conduct a page-by-page, section-by-section analysis. Be meticulous and ensure you don't miss any relevant information that might be scattered throughout the document. 
        
        Here's the complete content of the document in markdown format:
        
        {markdown_content}
        """
        
        # Create the message
        messages = [
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Process with Claude in streaming mode with thinking enabled
        thinking_text = ""
        answer_text = ""
        
        # Open files for saving outputs in real-time
        thinking_file = open("thinking_output.txt", "w", encoding="utf-8")
        answer_file = open("answer_output.txt", "w", encoding="utf-8")
        
        print("\nStreaming answer with extended thinking:")
        print("-----------------------------------------\n")
        
        # Initialize the stream with extended thinking
        with client.messages.stream(
            model="claude-3-7-sonnet-20250219",
            max_tokens=32000,
            thinking={
                "type": "enabled",
                "budget_tokens": 16000
            },
            messages=messages,
            system=system_message
        ) as stream:
            # Process events as they come
            for event in stream:
                if event.type == "content_block_start":
                    print(f"\nStarting {event.content_block.type} block...")
                    
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'type'):
                        if event.delta.type == "thinking_delta":
                            # Print thinking in gray
                            thinking_chunk = event.delta.thinking
                            thinking_text += thinking_chunk
                            thinking_file.write(thinking_chunk)
                            thinking_file.flush()
                            print(f"\033[90m{thinking_chunk}\033[0m", end="", flush=True)
                        
                        elif event.delta.type == "text_delta":
                            # Print answer in normal color
                            answer_chunk = event.delta.text
                            answer_text += answer_chunk
                            answer_file.write(answer_chunk)
                            answer_file.flush()
                            print(f"{answer_chunk}", end="", flush=True)
                
                elif event.type == "content_block_stop":
                    print("\nBlock complete.")
        
        # Close the files
        thinking_file.close()
        answer_file.close()
        
        print("\n\nStreaming complete.")
        print("Thinking process saved to thinking_output.txt")
        print("Answer saved to answer_output.txt")
        
        return answer_text
            
    except Exception as e:
        print(f"Error streaming answer with Claude: {e}")
        import traceback
        print(traceback.format_exc())
        return f"Error answering the question: {str(e)}"

def main():
    """Main function to run the document processing and Q&A."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Document understanding and Q&A tool with extended thinking')
    
    # Create argument groups for different modes
    group = parser.add_mutually_exclusive_group(required=True)
    
    # Mode 1: Document conversion (PDF to markdown)
    group.add_argument('--input', help='Path to input document (PDF, image) for conversion')
    parser.add_argument('--output', help='Path to output markdown file (used with --input)')
    
    # Mode 2: Document querying (from markdown)
    group.add_argument('--markdown', help='Path to markdown file for querying')
    parser.add_argument('--query', help='Question to ask about the document (used with --markdown)')
    
    args = parser.parse_args()
    
    # Mode 1: Document conversion
    if args.input:
        input_path = args.input
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            return
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}.md"
        
        # Initialize Mistral client for document understanding
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            print("Error: MISTRAL_API_KEY environment variable not set.")
            print("Please set the API key using: export MISTRAL_API_KEY='your_api_key'")
            return
        
        mistral_client = Mistral(api_key=mistral_api_key)
        print("Mistral client initialized successfully")
        
        # Process the document with Mistral's document understanding
        content = process_document_with_mistral(mistral_client, input_path)
        
        if not content or not content.strip():
            print("Error: No text content could be extracted from the document.")
            return
        
        # Save the markdown content
        save_markdown(content, output_path)
        print("\nDocument conversion:")
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print("Conversion completed successfully")
    
    # Mode 2: Document querying
    elif args.markdown:
        markdown_path = args.markdown
        
        # Check if markdown file exists
        if not os.path.exists(markdown_path):
            print(f"Error: Markdown file not found: {markdown_path}")
            return
        
        # Check if query is provided
        if not args.query:
            print("Error: --query is required when using --markdown")
            return
        
        # Read the markdown file
        markdown_content = read_markdown_file(markdown_path)
        if not markdown_content:
            return
        
        # Initialize Claude client for answering questions
        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not claude_api_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set.")
            print("Please set the API key using: export ANTHROPIC_API_KEY='your_api_key'")
            return
        
        claude_client = anthropic.Anthropic(api_key=claude_api_key)
        print("Claude client initialized successfully")
        
        # Stream the answer using Claude and the markdown content
        print("\nQuestion: " + args.query)
        stream_answer_with_claude(claude_client, markdown_content, args.query)

if __name__ == "__main__":
    main()