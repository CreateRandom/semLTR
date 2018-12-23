import api_config.apiKeys as apiKeys
import util.webHelpers as webHelpers
import requests
import objectpath
from util.general_util import compute_uuid_from_trec_id

chatnoirKey = apiKeys.keys['chatnoir']['key']
baseURL = apiKeys.keys['chatnoir']['base_url']

# Make a query, returning search results
# https://www.chatnoir.eu/api/v1/_search?apikey=key&query=hello%20world&index=cw09&size=1000&pretty
searchURL = baseURL + 'api/v1/_search?'
def makeQuery(query, n_results = 10, corpus = 'cw09'):
    json = webHelpers.get_json_from_url(searchURL +
                                 'apikey=' + chatnoirKey +
                                 '&query=' + query +
                                 '&index=' + corpus +
                                 '&size=' + str(n_results) + '&pretty')

    # we only care about the search results, not the meta information
    results = json['results']
    return results

# For a given query, get the list of trec ids retrieved
def get_chatnoir_ranking(query,n_results=10,corpus='cw09'):
    if not is_Chatnoir_online():
        raise RuntimeError('Could not reach ChatNoir server!')

    results = makeQuery(query,n_results,corpus)

    # retrieve the trec_ids from the results
    json_tree = objectpath.Tree(results)
    ranked_ids = list(json_tree.execute('$..trec_id'))

    return ranked_ids


def retrieve_word_constants_from_chatnoir(term, corpus='cw09'):
    # retrieve a single result for the query and request an explanation
    json = webHelpers.get_json_from_url(searchURL +
                                        'apikey=' + chatnoirKey +
                                        '&query=' + term +
                                        '&index=' + corpus +
                                        '&size=' + str(1) + '&pretty'+'&explain')
    # explanation json
    explanation = json['results'][0]['explanation']
    # TODO retrieve the idf scores for body, full_body and title from this json
    # This is how we get a list that contains the same kind of information for different parts of a document:
    all_document_sections_info = explanation['details'][0]['details'][0]['details'][0]['details']#[0]#['details'][0]['details'][0]
    sections_constants = {}
    # Now, loop over this information list to extract the information that belongs to each particular document section:
    for section_info in all_document_sections_info:
        # Extract the name of the document section (e.g. 'headings_lang','meta_desc_lang', 'body_lang' etc)
        section_name_description = section_info['description']
        section_name = section_name_description.split("weight(")[1].split(".")[0].replace("_lang","")
        # Now, extract the idf value of that section
        sections_idf = section_info['details'][0]['details'][0]['value']
        # We also want to store the avgFieldLength
        sections_avgFieldLength = section_info['details'][0]['details'][1]['details'][3]['value']
        # Store the information in the dict:
        sections_constants[section_name] = {'idf':sections_idf,'avgFieldLength': sections_avgFieldLength}
        del section_name_description,section_name,section_info,sections_idf,sections_avgFieldLength  # Just for readability during debugging sessions


    # headings_lang
    # meta_desc_lang
    # body_lang
    # full_body_lang
    # title_lang
    return sections_constants

# Input: a UUID we want to retrieve
# Output: the URL string of a query to retrieve that uuid
def get_chatnoir_url_for_uuid(uuid, corpus= 'cw09'):
    return retrievalURL + 'uuid=' + uuid + '&index=' + corpus + '&raw&plain'

# Retrieve the plain text HTML version of a document (represented by its uuid)
# https://www.chatnoir.eu/cache?uuid=7f9c78ec-91c1-5d25-af5b-43ce1ded5864&index=cw09&raw&plain
retrievalURL = baseURL + 'cache?'
def retrieve_HTML_from_UUID(uuid, corpus ='cw09'):
    return webHelpers.get_http_response_from_url(get_chatnoir_url_for_uuid(uuid,corpus))

# computes the internal ID from a single TREC ID
def retrieveHTML_from_TREC_ID(trec_id, corpus ='cw09'):
    uuid = compute_uuid_from_trec_id(trec_id)
    return retrieve_HTML_from_UUID(uuid,corpus)

def is_Chatnoir_online():
    try:
        import httplib
    except:
        import http.client as httplib

    conn = httplib.HTTPConnection("www.chatnoir.eu", timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False


# bulk retrieval for multiple ids
# returns a dict that contains a mapping from ids to retrieved HTML
def retrieveHTML_from_TREC_IDs(trec_id_list, corpus ='cw09', verbose=False):
    # check internet connection first
    if not is_Chatnoir_online():
        raise RuntimeError('Could not reach ChatNoir server!')

    if verbose: print('Starting session with ' + str(len(trec_id_list)) + ' items.')
    session = requests.Session()

    id_to_html = {}
    # for all ids to be retrieved
    for n_retrieved, trec_id in enumerate(trec_id_list):
        uuid = compute_uuid_from_trec_id(trec_id)
        url = get_chatnoir_url_for_uuid(uuid)
        # an HTML response
        response = session.get(url)
        # store the content of the response
        id_to_html[trec_id] = response.content.decode("utf8")
        if verbose:

            if (n_retrieved >  0 and n_retrieved % 100 == 0):
                print('Retrieved ' + str(n_retrieved) + ' documents.')

    return id_to_html