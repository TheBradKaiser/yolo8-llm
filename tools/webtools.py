import requests
from bs4 import BeautifulSoup
from googlesearch import search

def get_main_text(url:str) -> str:
    """
    Fetches the main text content of a webpage.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: The main text content of the webpage, or an error message if scraping fails.
    """
    try:
        # Send a GET request to the URL
        if "https://" not in url:
            if "http://" in url:
                url = url.replace('http://','https://')
            else:
                url = "https://"+url
            print(url)
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Attempt to find the main content of the page
        # Common HTML tags/ids/classes for main content
        main_content = soup.find('main')
        if not main_content:
            main_content = soup.find('div', {'id': 'content'})
        if not main_content:
            main_content = soup.find('div', {'class': 'main-content'})

        # Fallback to the entire body if main content is not found
        if not main_content:
            main_content = soup.find('body')

        # Extract and clean the text
        text = main_content.get_text(separator='\n', strip=True)
        return text

    except requests.exceptions.RequestException as e:
        return f"Error fetching the URL: {e}"
    except Exception as e:
        return f"Error parsing the page: {e}"

# print(get_main_text("https://yahoo.com"))


def make_api_call(url, method, params=None, query=None, data=None, headers=None, timeout=30):
    """
    Makes an API call to a given URL with specified parameters.

    Args:
        url (str): The API endpoint URL.
        method (str): HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE').
        params (dict, optional): URL parameters to append to the URL.
        query (dict, optional): Query parameters to include in the request.
        data (dict, optional): Data to send in the body of the request (for POST, PUT, etc.).
        headers (dict, optional): HTTP headers to include in the request.
        timeout (int, optional): Timeout for the request in seconds (default: 30).

    Returns:
        dict: A dictionary containing the response status code, headers, and JSON content (if available).

    Raises:
        requests.exceptions.RequestException: For network-related errors.
    """
    try:
        # Make the API call
        response = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            json=query,
            data=data,
            headers=headers,
            timeout=timeout
        )

        # Raise an error if the response status code is not successful (4xx or 5xx)
        response.raise_for_status()

        # Attempt to parse the response as JSON
        try:
            content = response.json()
        except ValueError:
            content = response.text

        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': content
        }

    except requests.exceptions.RequestException as e:
        return {
            'error': str(e),
            'status_code': None,
            'headers': None,
            'content': None
        }


def google_search(query, num_results=5):
    """
    Performs a Google search and returns summaries of the top results along with their links.

    Args:
        query (str): The search query.
        num_results (int): Number of results to return (default: 5).

    Returns:
        list: A list of dictionaries containing the title, link, and summary for each result.
    """
    results = []
    for url in search(query, num=num_results, stop=num_results):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the page title
            title = soup.title.string if soup.title else 'No title available'

            # Extract the meta description or first paragraph as summary
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            summary = meta_desc['content'] if meta_desc else None

            if not summary:
                # Use the first paragraph as a fallback summary
                first_paragraph = soup.find('p')
                summary = first_paragraph.text.strip() if first_paragraph else 'No summary available'

            results.append({
                'title': title,
                'link': url,
                'summary': summary
            })
        except requests.RequestException as e:
            results.append({
                'title': 'Error fetching page',
                'link': url,
                'summary': str(e)
            })

    return results