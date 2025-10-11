import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utlis import *
from tqdm import tqdm
from openai import OpenAI
import time

#from Models import *
#from prompt_generation import *
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_shot', type=str)
    return parser.parse_args()

def load_test_data(args):
    """Loads the test dataset based on the provided arguments."""
    dataset = load_dataset("zou-lab/MedCaseReasoning", "default") 
    dataset_test=dataset['test']
    
    return dataset_test
def process_file(data):
    """Process a single file for predictions and metrics calculation"""
    
    try:
        pred = data['prediction'].strip()
        pred = parse_generation(pred)
    except Exception:
        pred = ''
    
    gts = data['diagnostic_reasoning']
    #gts = [gt[1:-1].strip() if gt[0] == '(' and gt[-1] == ')' else gt for gt in gts]
    return pred, gts
def parse_generation(pred):
    """Parse generated answers based on rules"""
    for start_identifier in ['<think>']:
        if start_identifier in pred:
            pred = pred.split(start_identifier)[-1].strip()
            break

    for end_identifier in ['</think>']:
        if end_identifier in pred:
            pred = pred.split(end_identifier)[0].strip()
        break

    
    return pred.strip()
def generate_and_save_prompts(data_test,folder_path,  folder_path_in,num_shot,args):
    """Generates prompts, processes them in batches, and saves results."""
    batch_size = 4
    input_prompts = []
    file_paths = []
    samples = []
    
    for i in tqdm(range(len(data_test))):
        file_path = f'{folder_path}/{str(i)}.json'
        if os.path.exists(file_path):
            continue

        sample = data_test[i]
        file_path_in = folder_path_in + f'/{str(i)}.json'
        with open(file_path_in) as json_file:
            data_in = json.load(json_file)

        pred, gts = process_file(data_in)
        cur_prompt = my_generate_prompt_ICL(  
            pred, gts,num_shot
        )
        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= batch_size:
            run_one_batch_ICL(input_prompts, samples, file_paths)
            input_prompts, file_paths, samples = [], [], []

    if len(input_prompts) > 0:
        run_one_batch_ICL(input_prompts, samples, file_paths)
def my_generate_prompt_ICL(diagnosis,groundtruth,num_shot):
    '''
    Gnerate the prompt for the model

    args:
    story: the story, str
    Q: the question, str
    C: the candidates, list
    Q_type: the question type, str

    return:
    prompt: the generated prompt, str
    '''
    prompt_case = f"""
You are an experienced medical expert tasked with comparing diagnostic reasoning statements that support a given
diagnosis for a given patient case.
Your goal is to find supporting statements in the predicted diagnostic reasons that match the groundtruth
diagnostic reasons.
For each of the statements in Groundtruth Diagnostic Reasons, you need to find the statement or statements in the
Predicted Diagnostic Reasons that state the equivalent justification for the diagnosis.
For instance, if the groundtruth diagnostic reason is "The patient has a fever", and the predicted diagnostic
reason is "The patient has a fever due to a viral infection", then this is a match.
If the groundtruth diagnostic reason is "The patient has a fever", and the predicted diagnostic reason is "The
patient has a sore throat", then this is not a match.
Instructions:
1. Analyze each statement in Groundtruth Diagnostic Reasons.
2. For each statement in Groundtruth Diagnostic Reasons, find any matching statements in Predicted Diagnostic
Reasons.
3. Compute the recall, defined as the number of statements from the Groundtruth Diagnostic Reasons that appear in the Predicted Diagnostic Reasons, divided by the total number of statements in the Groundtruth Diagnostic Reasons. Wrap the recall value within the tags <answer> ... </answer>.
"""
    prompt = f"{prompt_case}\nPredicted Diagnostic Reasons:\n{diagnosis}\nGroundtruth Diagnostic Reasons:\n{groundtruth}\nYour analysis and final output:"

    return prompt

def my_generate_prompt_ICL_ori(diagnosis,groundtruth,num_shot):
    '''
    Gnerate the prompt for the model

    args:
    story: the story, str
    Q: the question, str
    C: the candidates, list
    Q_type: the question type, str

    return:
    prompt: the generated prompt, str
    '''
    prompt_case = f"""
You are an experienced medical expert tasked with comparing diagnostic reasoning statements that support a given
diagnosis for a given patient case.
Your goal is to find supporting statements in the predicted diagnostic reasons that match the groundtruth
diagnostic reasons.
For each of the statements in Groundtruth Diagnostic Reasons, you need to find the statement or statements in the
Predicted Diagnostic Reasons that state the equivalent justification for the diagnosis.
For instance, if the groundtruth diagnostic reason is "The patient has a fever", and the predicted diagnostic
reason is "The patient has a fever due to a viral infection", then this is a match.
If the groundtruth diagnostic reason is "The patient has a fever", and the predicted diagnostic reason is "The
patient has a sore throat", then this is not a match.
Instructions:
1. Analyze each statement in Groundtruth Diagnostic Reasons.
2. For each statement in Groundtruth Diagnostic Reasons, find any matching statements in Predicted Diagnostic
Reasons.
3. Create a JSON object with the following structure:
- The main key should be "matching_dict"
- Each key within "matching_dict" should be a number representing a statement from Groundtruth Diagnostic
Reasons
- The value for each key should be a list of matching statements from Predicted Diagnostic Reasons
- If there are no matches for a statement, use an empty array
Before providing your final output, wrap your analysis inside <diagnostic_comparison> tags:
1. List all statements from Groundtruth Diagnostic Reasons and Predicted Diagnostic Reasons.
2. For each statement in Groundtruth Diagnostic Reasons, consider potential matches from Predicted Diagnostic
Reasons:
- List pros and cons for each potential match
- It’s OK for this section to be quite long
3. Summarize your final matching decisions
4. In the JSON output, only include the statements that are in the Predicted Diagnostic Reasons.
5. In the JSON output, the statements should appear exactly as they are in the Predicted Diagnostic Reasons,
verbatim, letter for letter. Do not modify the statements in any way, such as rewording them, adding
punctuation, quotes, etc.
Wrap your JSON output in ‘‘‘json tags.
Example of the required JSON structure:

```json
{{
"matching_dict": {{
"1": [],
"2": ["Matching statement 1", "Matching statement 2"],
"3": ["Matching statement 3"]
}}
}}
"""
    prompt = f"{prompt_case}\nPredicted Diagnostic Reasons:\n{diagnosis}\nGroundtruth Diagnostic Reasons:\n{groundtruth}\nYour analysis and final output:"

    return prompt

def run_one_batch_ICL(input_prompts, samples, file_paths, max_new_tokens=512):
    '''
    Generate the completion for one batch of input prompts

    args:
    input_prompts: the input prompts, list
    samples: the samples, list
    file_paths: the file paths to save the results, list
    max_new_tokens: the maximum new tokens for the completion

    return:
    None
    '''
   
    client = OpenAI(api_key="", base_url="https://api.deepseek.com")

    for j in range(len(input_prompts)):
        prompt = input_prompts[j]
        cur_sample = samples[j]

        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )

            prediction = response.choices[0].message.content.strip()
            cur_sample.update({'prediction': prediction})

            with open(file_paths[j], 'w', encoding='utf-8') as json_file:
                json.dump(cur_sample, json_file, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[Error] Failed to generate for input {j}: {e}")


    return



def main():
    #time.sleep(3*60*60)
    args = parse_args()
    dataset_name="MedReason"
    # Load dataset and initialize variables
    data_test = load_test_data(args)
    
    
    #folder_path = f'../results/{dataset_name}_{args.model}_{args.num_shot}shot'
    folder_path_in = f'../results/{dataset_name}_{args.model}_10shot_deep_research_cleaned2'
    folder_path = f'../results/{dataset_name}_{args.model}_10shot_deep_research_cleaned2_eval1'
    #folder_path_in = f'../results/{dataset_name}_{args.model}_10shot'
    #folder_path = f'../results/{dataset_name}_{args.model}_10shot_eval2'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    # Print example prompts if specified
    

    # Initialize model and tokenizer
    #model, tokenizer = initialize_model_and_tokenizer(model_name)

    # Generate and save prompts
    generate_and_save_prompts(data_test,folder_path,folder_path_in, args.num_shot,args)


if __name__ == "__main__":
    main()
