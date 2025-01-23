import json
import sys

def convert_json_to_jsonl(input_file, output_file):
    """
    Convert a JSON file containing an array of objects to JSONL format.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSONL file
    """
    try:
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the input is a list/array
        if not isinstance(data, list):
            data = [data]  # Convert single object to list
        
        # Write each object as a separate line in JSONL format
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
                
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input.json output.jsonl")
    else:
        convert_json_to_jsonl(sys.argv[1], sys.argv[2])