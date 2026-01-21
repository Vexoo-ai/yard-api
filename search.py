import os
import re
import asyncio
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from aiohttp import ClientSession
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRanker
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse
# from serpapi import GoogleSearch

load_dotenv()

# Initialize SentenceTransformer model globally
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute TLD scores
TLD_SCORES = {
    'com': 0.7, 'org': 0.6, 'net': 0.5, 'edu': 0.8, 'gov': 0.9,
    'io': 0.4, 'co': 0.5, 'ai': 0.3, 'app': 0.3
}

def get_domain_authority(domain):
    tld = domain.split('.')[-1]
    base_score = TLD_SCORES.get(tld, 0.3)
    length_factor = max(0, (20 - len(domain)) / 20)
    return min(1.0, base_score + (length_factor * 0.3))

def get_content_freshness(result):
    current_time = datetime.now()
    date_str = result.get('date') or result.get('snippet', '')
    
    date_patterns = [
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b',
        r'\b(\d{4}-\d{2}-\d{2})\b'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                date = datetime.strptime(match.group(1), '%d %b %Y')
            except ValueError:
                try:
                    date = datetime.strptime(match.group(1), '%Y-%m-%d')
                except ValueError:
                    continue
            
            days_old = (current_time - date).days
            return max(0, 1 - (days_old / 365))
    
    position = result.get('position', 0)
    return max(0, 1 - (position / 20))

async def fetch_search_results(session, params):
    async with session.get('https://serpapi.com/search', params=params) as response:
        return await response.json()

async def call_search_engines(query):
    serpapi_api_key = os.getenv('SERPAPI_API_KEY')
    
    params = {
        "api_key": serpapi_api_key,
        "q": query,
        "hl": "en",
        "gl": "us",
    }
    
    async with ClientSession() as session:
        tasks = [
            fetch_search_results(session, {**params, "engine": engine})
            for engine in ["google", "bing", "duckduckgo"]
        ]
        results = await asyncio.gather(*tasks)
    
    all_results = {
        engine: result.get('organic_results', [])
        for engine, result in zip(["google", "bing", "duckduckgo"], results)
    }
    
    return process_and_rank_results(all_results, query)

def process_and_rank_results(all_results, query):
    combined_results = [
        {**result, 'engine': engine}
        for engine, results in all_results.items()
        for result in results
    ]
    
    query_embedding = sentence_model.encode([query])[0]
    
    features = []
    for result in combined_results:
        snippet = result.get('snippet', '')
        title = result.get('title', '')
        url = result.get('link', '')
        
        snippet_embedding = sentence_model.encode([snippet])[0]
        title_embedding = sentence_model.encode([title])[0]
        
        semantic_similarity_snippet = cosine_similarity([query_embedding], [snippet_embedding])[0][0]
        semantic_similarity_title = cosine_similarity([query_embedding], [title_embedding])[0][0]
        
        domain = urlparse(url).netloc
        domain_authority = get_domain_authority(domain)
        
        features.append([
            result.get('position', 0),
            len(snippet),
            snippet.count(query),
            semantic_similarity_snippet,
            semantic_similarity_title,
            domain_authority,
            get_content_freshness(result),
            int(result['engine'] == 'google'),
            int(result['engine'] == 'bing'),
            int(result['engine'] == 'duckduckgo')
        ])
    
    X = MinMaxScaler().fit_transform(np.array(features))
    
    y = np.arange(len(features))[::-1]
    y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).ravel()
    y = (y * 30).astype(int)
    
    group = [len(features)]
    
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="dart",
        n_estimators=100,
        importance_type="gain",
        max_position=31
    )
    model.fit(X, y, group=group)
    
    scores = model.predict(X)
    
    sorted_indices = np.argsort(scores)[::-1]
    ranked_results = [combined_results[i] for i in sorted_indices]
    
    diverse_results = ensure_diversity(ranked_results)
    
    return {'organic_results': diverse_results}

def ensure_diversity(results, diversity_threshold=0.3):
    diverse_results = []
    domains_included = set()
    
    for result in results:
        domain = urlparse(result.get('link', '')).netloc
        if domain not in domains_included or len(diverse_results) < 5:
            diverse_results.append(result)
            domains_included.add(domain)
        
        if len(diverse_results) >= 10:
            break
    
    return diverse_results

def format_search_results(search_data):
    formatted_results = []
    for result in search_data.get('organic_results', []):
        displayed_link = result.get('displayed_link', '')
        source = None
        if displayed_link:
            if '://' in displayed_link:
                displayed_link = displayed_link.split('://', 1)[-1]
            source = displayed_link.split('/')[0]

        formatted_result = {
            'source': source,
            'date': None,
            'title': result.get('title', ''),
            'snippet': result.get('snippet', ''),
            'highlight': result.get('snippet_highlighted_words', ''),
            'engine': result.get('engine', ''),
            'link': result.get('link', '')
        }
        formatted_results.append(formatted_result)
    return formatted_results

# async def fetch_google_scholar_results(query, num_results=40, sort_by='date'):
#     serpapi_api_key = os.getenv('SERPAPI_API_KEY')
    
#     params = {
#         "api_key": serpapi_api_key,
#         "engine": "google_scholar",
#         "q": query,
#         "num": num_results,
#         "sort": sort_by,
#         "as_ylo": datetime.now().year - 5
#     }
    
#     search = GoogleSearch(params)
#     results = search.get_dict()
#     return results.get("organic_results", [])
