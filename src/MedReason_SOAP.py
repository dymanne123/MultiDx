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

def generate_and_save_prompts(data_test, folder_path, num_shot,args):
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
        cur_prompt = my_generate_prompt_structured(  
            sample['case_prompt'],num_shot
        )
        input_prompts.append(cur_prompt)
        samples.append(sample)
        file_paths.append(file_path)

        if len(input_prompts) >= batch_size:
            run_one_batch_ICL(input_prompts, samples, file_paths)
            input_prompts, file_paths, samples = [], [], []

    if len(input_prompts) > 0:
        run_one_batch_ICL(input_prompts, samples, file_paths)

def my_generate_prompt_ICL(case_prompt,num_shot):
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
    file_path = f'../materials/MedReason/prompt_examples_input_10.txt'
    file_path_out = f'../materials/MedReason/prompt_examples_output.txt'
    file_path_example = f'../materials/MedReason/example_{num_shot}.txt'

    with open(file_path) as txt_file:
        prompt_in = txt_file.read()
    with open(file_path_out) as txt_file:
        prompt_out = txt_file.read()

    with open(file_path_example) as txt_file:
        prompt_example = txt_file.read()
    #prompt = f"{prompt_in}\nExample:\n\n{prompt_example}\n{case_prompt}\n{prompt_out}"
    prompt = f"{prompt_in}\n{case_prompt}\n{prompt_out}"

    return prompt
def my_generate_prompt_structured(case_prompt,num_shot):
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
    file_path = f'../materials/MedReason/prompt_SOAP1.txt'
    file_path_out = f'../materials/MedReason/prompt_examples_output.txt'
    file_path_example = f'../materials/MedReason/example_{num_shot}.txt'

    with open(file_path) as txt_file:
        prompt_in = txt_file.read()
    with open(file_path_out) as txt_file:
        prompt_out = txt_file.read()

    with open(file_path_example) as txt_file:
        prompt_example = txt_file.read()
    #prompt = f"{prompt_in}\nExample:\n\n{prompt_example}\n{case_prompt}\n{prompt_out}"
    prompt = f"{prompt_in}\n{case_prompt}\n{prompt_out}"
    #prompt = f"{prompt_in}\n{case_prompt}"

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

def run_one_batch_refine(input_prompts, samples, file_paths, max_new_tokens=512):
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
    max_iters = 3
    for j in range(len(input_prompts)):
        prompt = input_prompts[j]
        cur_sample = samples[j]

        try:
            messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(model="deepseek-reasoner", messages=messages)
            y0 = response.choices[0].message.content.strip()

            history = [{"output": y0, "feedback": ""}]
            yt = y0  # Current output

            ########################
            # Step 2~N: Feedback → Refine
            ########################
            for t in range(max_iters):
                # Generate Feedback
                fb_prompt = f"""Please provide actionable and specific feedback to improve the following output.If the output is already good, you can say \"no improvements needed\" or \"looks good\".:
                                Input: {prompt}
                                Output: {yt}
                                Feedback:"""

                fb_response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are a critical feedback generator."},
                        {"role": "user", "content": fb_prompt}
                    ]
                )
                feedback = fb_response.choices[0].message.content.strip()

                # Stopping condition
                if "no improvements needed" in feedback.lower() or "looks good" in feedback.lower():
                    break

                # Refine the output
                refine_prompt = f"""You are an assistant that improves answers based on user feedback.
                                    Input: {prompt}
                                    Previous Output: {yt}
                                    Feedback: {feedback}
                                    Improved Output:"""

                refine_response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that refines outputs."},
                        {"role": "user", "content": refine_prompt}
                    ]
                )
                yt = refine_response.choices[0].message.content.strip()
                history.append({"output": yt, "feedback": feedback})

            ########################
            # Save Final Output
            ########################
            cur_sample.update({
                'prediction': yt,
                'self_refine_history': history  # Optional: saves all iterations
            })

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
    folder_path = f'../results/{dataset_name}_{args.model}_10shot_SOAP2'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    # Print example prompts if specified
    

    # Initialize model and tokenizer
    #model, tokenizer = initialize_model_and_tokenizer(model_name)

    # Generate and save prompts
    generate_and_save_prompts(data_test, folder_path, args.num_shot,args)


if __name__ == "__main__":
    main()
