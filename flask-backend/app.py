from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import pickle
import requests  # Add this import statement at the top of your file


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load preprocessed data and model
model = Doc2Vec.load('doc2vec.model')
papers = pd.read_pickle('papers.pkl')
with open('citation_similarity.pkl', 'rb') as f:
    citation_similarity = pickle.load(f)

# Clean the abstracts column (if needed)
papers['abstract'] = papers['abstract'].fillna('')

# Extract abstracts and paper IDs
abstracts = papers['abstract'].tolist()

# Function to recommend similar abstracts
def recommend_papers(doc_id, top_n=5):
    inferred_vector = model.infer_vector(word_tokenize(abstracts[doc_id].lower()))
    content_sims = model.dv.most_similar([inferred_vector], topn=len(papers))
    content_sims_dict = {int(tag): sim for tag, sim in content_sims}

    max_content_sim = max(content_sims_dict.values())
    content_sims_dict = {k: v / max_content_sim for k, v in content_sims_dict.items()}

    citation_sims = citation_similarity[doc_id]
    citation_sims_dict = {i: citation_sims[i] for i in range(len(papers))}

    max_citation_sim = max(citation_sims_dict.values())
    if max_citation_sim > 0:
        citation_sims_dict = {k: v / max_citation_sim for k, v in citation_sims_dict.items()}
    else:
        citation_sims_dict = {k: 0 for k in citation_sims_dict}

    combined_sims = {i: content_sims_dict.get(i, 0) * 0.7 + citation_sims_dict.get(i, 0) * 0.3 for i in range(len(papers))}

    recommended_abstracts = sorted(combined_sims.items(), key=lambda item: item[1], reverse=True)
    recommended_abstracts = [(idx, sim) for idx, sim in recommended_abstracts if idx != doc_id][:top_n]

    return recommended_abstracts


def get_common_citation_ratio(original_doc_id, recommended_doc_id):
    original_cited_papers = papers.loc[original_doc_id, 'citations']
    recommended_cited_papers = papers.loc[recommended_doc_id, 'citations']
    
    if not recommended_cited_papers:
        return 0
    
    common_with_rec = sum(1 for cited_id in recommended_cited_papers if cited_id in original_cited_papers)
    total_rec_citations = len(recommended_cited_papers)
    
    return common_with_rec / total_rec_citations


def get_paper_from_semantic_scholar(paper_title):
    search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={paper_title}&limit=1&fields=title,authors,abstract"
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        
        data = response.json()
        if 'data' in data and data['data']:
            paper_info = data['data'][0]
            return paper_info
        else:
            return None
    except requests.RequestException as e:
        print(f"Error fetching data from Semantic Scholar: {e}")
        return None


# def add_recommended_to_citations(doc_id, recommendations, lower_threshold=0.25,upper_threshold=0.75):
#     updated = False
#     for idx, sim in recommendations:
#         common_ratio = get_common_citation_ratio(doc_id, idx)
        
#         if common_ratio < lower_threshold and idx not in papers.loc[doc_id, 'citations']:
#             papers.at[doc_id, 'citations'].append(idx)  # Add to original paper citations
#             updated = True
#         if common_ratio > upper_threshold and idx not in papers.loc[doc_id, 'citations']:
#             papers.at[doc_id, 'citations'].append(idx)  # Add to original paper citations
#             updated = True
    
#     if updated:
#         # Save the updated dataframe back to the file to persist the changes
#         papers.to_pickle('papers_updated.pkl')



# def add_recommended_to_citations(doc_id, recommendations, lower_threshold=0.25, upper_threshold=0.75):
#     updated = False
    
#     # Ensure single row is accessed, add a debug print
#     if doc_id not in papers.index:
#         print(f"Document ID {doc_id} not found in the index.")
#         return

#     # Ensure the citations field is a list
#     citations = papers.loc[doc_id, 'citations']
#     print(f"Before conversion, citations for doc_id {doc_id}: {citations}, type: {type(citations)}")
    
#     # Convert citations to list if it's a string
#     if isinstance(citations, str):
#         try:
#             citations = eval(citations)
#         except Exception as e:
#             print(f"Failed to evaluate citations for doc_id {doc_id}: {e}")
#             return
    
#     if not isinstance(citations, list):
#         citations = []

#     for idx, sim in recommendations:
#         common_ratio = get_common_citation_ratio(doc_id, idx)
        
#         if common_ratio < lower_threshold and idx not in citations:
#             citations.append(idx)
#             updated = True
#         if common_ratio > upper_threshold and idx not in citations:
#             citations.append(idx)
#             updated = True

#     if updated:
#         # Update the dataframe with the modified list
#         papers.at[doc_id, 'citations'] = citations
#         print(f"Updated citations for doc_id {doc_id}: {citations}")

#         # Save the updated dataframe back to the file to persist the changes
#         papers.to_pickle('papers_updated.pkl')


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    paper_title = data.get('paper_title')

    doc_id = next((i for i, paper in papers.iterrows() if paper['title'].lower() == paper_title.lower()), None)

    if doc_id is None:
        # If not found, try to fetch the paper using Semantic Scholar API
        external_paper = get_paper_from_semantic_scholar(paper_title)
        if external_paper is None:
            return jsonify({"error": "Paper not found in local dataset or Semantic Scholar"}), 404

        # Use the fetched abstract for similarity calculation
        external_abstract = external_paper.get('abstract', '')
        if not external_abstract:
            return jsonify({"error": "Abstract not found for the requested paper"}), 404
        

        external_authors = [author['name'] for author in external_paper.get('authors', [])]
        if not external_authors:
            external_authors = ["Unknown author"]

        # Compute similarity using the retrieved abstract
        inferred_vector = model.infer_vector(word_tokenize(external_abstract.lower()))
        content_sims = model.dv.most_similar([inferred_vector], topn=len(papers))
        content_sims_dict = {int(tag): sim for tag, sim in content_sims}

        max_content_sim = max(content_sims_dict.values())
        content_sims_dict = {k: v / max_content_sim for k, v in content_sims_dict.items()}

        # Use only content-based similarity as there's no citation info for external papers
        combined_sims = {i: content_sims_dict.get(i, 0) * 0.7 for i in range(len(papers))}

        recommended_abstracts = sorted(combined_sims.items(), key=lambda item: item[1], reverse=True)
        recommended_abstracts = [(idx, sim) for idx, sim in recommended_abstracts][:5]

        # Return response with recommendations for external paper
        response = {
            "paper_title": external_paper['title'],
            "authors": external_authors,  
            "abstract": external_abstract,
            "recommendations": []
        }
        
        for idx, sim in recommended_abstracts:
            response['recommendations'].append({
                "paper_title": papers.loc[idx, 'title'],
                "authors": papers.loc[idx, 'authors'],
                "abstract": papers.loc[idx, 'abstract'],
                "similarity": sim
            })
            
    else:
        # If found in the dataset, use the regular recommendation process
        recommendations = recommend_papers(doc_id)
        #add_recommended_to_citations(doc_id, recommendations)

        response = {
            "paper_title": papers.loc[doc_id, 'title'],
            "authors": papers.loc[doc_id, 'authors'],
            "abstract": papers.loc[doc_id, 'abstract'],
            "recommendations": []
        }

        for idx, sim in recommendations:
            response['recommendations'].append({
                "paper_title": papers.loc[idx, 'title'],
                "authors": papers.loc[idx, 'authors'],
                "abstract": papers.loc[idx, 'abstract'],
                "similarity": sim
            })
    return jsonify(response)


if __name__ == '__main__':
    app.run(port=6000, debug=True)


import os
print("Current working directory:", os.getcwd())

import os
print("Files in directory:", os.listdir())




