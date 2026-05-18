from datasets import load_dataset
import pandas as pd

from utils import write_jsonl


# Define a list of math and STEM-related keywords
# keywords_expanded = [
#     # Mathematics
#     "algebra", "geometry", "calculus", "statistics", "probability", "theorem", "proof", "equation", 
#     "integral", "derivative", "matrix", "vector", "graph", "function", "complex number", "real number", 
#     "imaginary number", "differential", "fraction", "decimal", "percent", "logarithm", 
#     "sequence", "sum", "product", "difference", "quotient", "prime", "composite", "factorial", "binomial", 
#     "polynomial", "exponential", "pi", "euler", "infinity", "limit", "derivative", "integral", 
#     "differential equation", "linear algebra", "set theory", "group theory", "ring theory", "field theory", 
#     "number theory", "combinatorics", "topology", "measure theory", "game theory", "cryptology", 
#     "algorithm", "computation","percentage","calculation"
    
#     # General STEM
#     "science", "technology", "engineering", "physics", "chemistry", "biology", "computer science", 
#     "information technology", "environmental", "aerospace", 
#     "biomedical", "chemical", "robotics", "AI", "artificial intelligence", 
#     "machine learning", "deep learning", "neural network", "algorithm", "programming", "coding", 
#     "software", "hardware", "network", "database", "security", "cybersecurity", "blockchain", 
#     "virtual reality", "augmented reality", "quantum", "nanotechnology", "biotechnology", "genetics", 
#     "genomics", "solar", "wind", "hydro", 
#     "nuclear", "fossil fuel", "carbon", "greenhouse gas", "pollution", "conservation", "biodiversity", 
#     "ecosystem", "species", "evolution", "cell", "molecule", "atom", "particle", "quantum", "gravity", 
#     "relativity", "momentum", "velocity", "acceleration", "mass", "heat", "light", "sound", "electricity", "magnetism",
# ]

data = load_dataset('metaeval/reclor')
data = pd.concat([pd.DataFrame(data[subset_name]) for subset_name in ("train", "validation")])

# Function for update
def format_question(data_entry):
    context = data_entry['context']
    question = data_entry['question']
    answers = data_entry['answers']
    label = data_entry['label']

    formatted_question = f"{question}\n\n{context}\n\nOptions:"
    for i, ans in enumerate(answers):
        formatted_question += "\n" + chr(65+i) + ": " + ans  

    # Create the formatted answer string
    formatted_answer = chr(65+label)

    return {"condition": "direct", "instruction": formatted_question, "response": formatted_answer}

reclor_data = [format_question(entry) for idx, entry in data.iterrows()]

# Filter the data based on the presence of any of these keywords in the instruction field
# filtered_data_keywords_reclor = [item for item in reclor_data if any(re.search(r'\b' + keyword + r'\b', item['instruction'].lower()) for keyword in keywords_expanded)]
# print('Number of examples kept: ',len(filtered_data_keywords_reclor))

# save to json file
write_jsonl(f"data/Platypus/reclor.jsonl", reclor_data)
