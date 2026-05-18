from datasets import load_dataset
import pandas as pd

from utils import write_jsonl


dataset = load_dataset('openbookqa', name='additional')

# put data in a dataframe
df_train = pd.DataFrame(dataset['train'])
df_val = pd.DataFrame(dataset['validation'])

# Put the dataframes into a single dataframe
df = pd.concat([df_train, df_val])
df.head()

# Convert the choices into multiple choice format
df['choices'] = df['choices'].apply(lambda x: [x['text'][i] for i in range(len(x['text']))])

# Start with an empty list to hold all the new JSON objects
json_objects_updated = []

# For each row in the dataframe
for idx, row in df.iterrows():
    # Parse the choices string into a list
    choices = row['choices']
    
    # Format the choices with alphabetic indicators
    formatted_choices = '\n'.join([f'\n{chr(65+i)}: {choice}' if i == 0 else f'{chr(65+i)}: {choice}' for i, choice in enumerate(choices)])
    
    # Combine the question stem with the formatted choices
    instruction = f"Based on the given fact, which of the following option is the correct answer to the question?\n\n{row['question_stem']} {formatted_choices}\n\nFact: {row['fact1']}"
    
    # Get the correct answer based on the answer key
    correct_answer = choices[ord(row['answerKey']) - 65]
    
    # Format the output with a random answer prefix from the updated list, the correct answer key, and the correct answer
    output = row['answerKey']
    
    # Create the JSON object and append it to the list
    json_objects_updated.append({
        "condition": "direct",
        "instruction": instruction,
        "response": output
    })

# Create a JSON file with the updated JSON objects, with an indent of 1 for readability
write_jsonl(f"data/Platypus/openbookqa.jsonl", json_objects_updated)
