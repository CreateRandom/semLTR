import json
import requests
import requests_cache
# basic caching for all requests made here
requests_cache.install_cache('semLTR_web_cache')

# A function to get a HTTP response from a URL using the requests library.
# INPUT
# url       the input URL given in the form of a string
# OUTPUT
# content   the HTTP response as a string
def get_http_response_from_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content

# A function to retrieve a web response from a given URL and parse it as a JSON.
# This obviously only works if the URL endpoint returns valid JSON in the first place.
# INPUT
# url       the input URL (string)
# OUTPUT
# js        the parsed JSON as a Python dict
def get_json_from_url(url):
    content = get_http_response_from_url(url)
    js = json.loads(content)
    return js




