import json
from tqdm import tqdm

def count_empty_text_lines(jsonl_file, output_file):
    empty_text_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as file, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(file):
            data = json.loads(line)
            # Check if "text" field exists and is either empty or only whitespace
            if 'text' not in data:
                raise KeyError("The 'text' field is missing in the input data.")
            
            if data['text'] is None or not data['text'].strip():
                empty_text_count += 1
            else:
                # Write the valid line (with non-empty "text" field) to the output file
                outfile.write(line)
    
    return empty_text_count

# Example usage
jsonl_file = '/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/data/cosmopedia.jsonl'
output_file = '/mnt/synthdatastore/agoswami/models_04_postlaborday/my_phi35_pretrain_trials_0920/data/cosmopedia_v2.jsonl'

empty_text_lines = count_empty_text_lines(jsonl_file, output_file)
print(f"Number of lines with empty 'text' field: {empty_text_lines}")
