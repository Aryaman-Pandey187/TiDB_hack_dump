import requests
import json

def search_pubmed(query, num_results=5):
    """Search PubMed for articles and return summaries with calculated relevance (simple keyword count)."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': num_results,
        'retmode': 'json'
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return {"error": "PubMed search failed"}
    data = response.json()
    ids = data['esearchresult']['idlist']
    
    summaries = []
    for id in ids:
        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id}&retmode=json"
        summary_response = requests.get(summary_url)
        if summary_response.status_code == 200:
            summary_data = summary_response.json()
            title = summary_data['result'][id]['title']
            # Simple 'relevance' calculation: keyword count in title
            relevance = title.lower().count(query.lower())
            summaries.append({'id': id, 'title': title, 'relevance': relevance})
    return summaries

def search_rxnorm(drug_name):
    """Map drug to RxNorm codes and compute therapeutic classes."""
    base_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
    params = {'name': drug_name}
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return {"error": "RxNorm search failed"}
    data = response.json()
    rxcui = data.get('idGroup', {}).get('rxnormId', [None])[0]
    if rxcui:
        info_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
        info_response = requests.get(info_url)
        if info_response.status_code == 200:
            info_data = info_response.json()
            properties = info_data.get('properties', {})
            # 'Compute' therapeutic class (simple extraction)
            classes = properties.get('therapeuticClasses', [])
            return {'rxcui': rxcui, 'properties': properties, 'therapeutic_classes_count': len(classes)}
    return {"error": "No RxCUI found"}

def search_mesh(term):
    """Link term to MeSH headings and calculate match count."""
    base_url = "https://meshb.nlm.nih.gov/api/search/term"
    params = {'term': term}
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return {"error": "MeSH search failed"}
    data = response.json()
    # Calculate 'match count' (simple length)
    match_count = len(data)
    return {'terms': data, 'match_count': match_count}

def search_openfda(drug_name, limit=5):
    """Fetch OpenFDA adverse events and calculate frequency stats."""
    base_url = "https://api.fda.gov/drug/event.json"
    params = {'search': f'patient.drug.openfda.brand_name:"{drug_name}"', 'limit': limit}
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return {"error": "OpenFDA search failed"}
    data = response.json()
    if data['meta']['results']['total'] > 0:
        events = data['results']
        # Calculate 'frequency' stats (e.g., count of unique reactions)
        reactions = [reaction for event in events for reaction in event.get('patient', {}).get('reaction', [])]
        unique_reactions = set(reaction['reactionmeddrapt'] for reaction in reactions if 'reactionmeddrapt' in reaction)
        return {'events': events, 'unique_reactions_count': len(unique_reactions)}
    return {"error": "No OpenFDA data found"}