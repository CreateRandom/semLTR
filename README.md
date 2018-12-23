# What it is
* Using semantic features for learning-to-rank in Text Retrieval. 
* Made for a course project of the 2018/2019 course [Information Retrieval](https://www.ru.nl/prospectus/socsci/courses-osiris/ai/nwi-i00041-information-retrieval/) at Radboud University. 
* Built atop the [ChatNoir web search engine](https://www.chatnoir.eu/) developed by [University of Weimar researchers](https://webis.de/).
# What it does
* For a set of reference queries, retrieves the 100 top results from ChatNoir.
* Re-ranks them using a machine-learning model trained on hand-labelled relevant documents.
* Compares different query-document representations for this purpose, including word2vec similarity and a new feature representation based on phrasal semantics.
# How to run it for yourself
* Apply for a ChatNoir API key and once you have it, add it to [this file](api_config/apiKeys.py).
* Download the TREC web track adhoc qrels and queries for 2010, 2011 and 2012 from the [TREC page](https://trec.nist.gov/data/webmain.html) and put them into the respective paths, e.g. TREC 2010 should go into [here](trec_data/trec_2010).
* Download the pre-trained Google News word embeddings from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit), unzip them and add them to [this folder](pretrained).
* You're good to go, run [the entry script](main_logic/run.py).
