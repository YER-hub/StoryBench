import json
import base64
import time
import os
import pandas as pd
from openai import OpenAI

# Proxy server address and port
# This is set to a local proxy (e.g., for VPN, network debugging)
proxy_url = 'http://127.0.0.1'
proxy_port = '7897'
os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

# Create and return an OpenAI client instance with the given API key and custom base URL
def create_client(api_key):
    return OpenAI(api_key=api_key, base_url="https://api.bltcy.ai/v1")

# Convert a local image file to a Base64-encoded string
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image: {str(e)}")
        return None

# Call the API using the OpenAI SDK
def query_model(image_base64, question, api_key, timeout=30):
    client = create_client(api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }}
                    ]
                }
            ],
            max_tokens=400,
            stream=False,
            timeout=timeout
        )
        return json.loads(response.json())
    except Exception as e:
        return {'error': str(e)}

def query_model_with_retry(image_base64, question, api_key, max_retries=5, timeout=40):
    for attempt in range(max_retries):
        try:
            response = query_model(image_base64, question, api_key, timeout)
            
            if 'error' in response:
                error_msg = str(response['error'])

                if (
                    "无可用渠道" in error_msg
                    or "Access denied due to invalid subscription key" in error_msg
                ):
                    print(f"🚫 Critical API Error detected: {error_msg}")
                    print("⏳ Sleeping for 5 minutes before retrying...")
                    time.sleep(5 * 60)  
                    continue

                print(f"API Error: {error_msg}, Retry {attempt+1}/{max_retries}")
                time.sleep(2 ** attempt + 1)
                continue

            return response

        except Exception as e:
            print(f"Request failed: {str(e)}, Retry {attempt+1}/{max_retries}")
            time.sleep(2 ** attempt + 1)

    return {'choices': [{'message': {'content': 'API_ERROR'}}]}

def load_prompts(json_path):
    try:
        with open(json_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading prompts: {str(e)}")
        return {}


def process_images_and_questions(prompts_data, image_dir, api_key, output_csv, delay=5):
    checkpoint_file = 'processing_checkpoint-DALL-E3-Two.json'
    partial_file = 'partial_results-DALL-E3-Two.csv'
    processed_images = set()
    results = []

    # Load checkpoint (names of already processed images)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                processed_images = set(json.load(f))
        except:
            print("Checkpoint file corrupted, starting fresh")

    # Load previously saved partial results
    if os.path.exists(partial_file):
        try:
            previous_results = pd.read_csv(partial_file).to_dict('records')
            results.extend(previous_results)
            for row in previous_results:
                if 'Image' in row:
                    processed_images.add(row['Image'])
            print(f"Resumed from {len(previous_results)} previous records.")
        except:
            print("Partial results file corrupted, starting fresh.")
    for category, prompt_list in prompts_data.items():
        for idx, prompt_data in enumerate(prompt_list):
            image_name = f"{category}_{idx + 1}.png"
            image_path = os.path.join(image_dir, image_name)
            print(f"\nProcessing image: {image_name}")

            if image_name in processed_images:
                print(f"Skipping already processed: {image_name}")
                continue

            if not os.path.exists(image_path):
                print(f"Missing image: {image_path}")
                continue

            image_base64 = image_to_base64(image_path)
            if not image_base64:
                continue

            answer_row = {
                'Category': category,
                'Image': image_name,
                'Prompt': prompt_data['prompt']
            }

            # Only Level 2
            for level, level_questions in prompt_data['questions'].items():
                if level != "Level 1":
                    continue
                for q_idx, question in enumerate(level_questions):
                    instruction = (
                                "Now,please score based on how well the content of the image matches the description in the question. The scoring criteria are as follows:\n "
                                "2 point for fully matching (the image content completely aligns with the question description, with no ambiguity or deviation),\n "
                                "1 point for partially matching (the image includes elements related to the question, but with some minor discrepancies in details or aspects), \n"
                                "0 point for not matching at all (the image does not show what is described in the question, or is completely inconsistent). \n"
                                "Please provide only the score, without any additional explanation.for example:0,1,2"
                            ) 
                    full_question = f"{question} {instruction}"
                    response = query_model_with_retry(image_base64, full_question, api_key)
                    answer = response.get('choices', [{}])[0].get('message', {}).get('content', 'API_ERROR')
                    print(f"Q{q_idx+1}: {question}")
                    print(f"Assistant: {answer}")
                    col_name = f"{level}_Q{q_idx+1}"
                    answer_row[col_name] = answer
                    time.sleep(max(delay, 1))

            results.append(answer_row)
            processed_images.add(image_name)

            # Write partial CSV and checkpoint
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump(list(processed_images), f)

                pd.DataFrame([answer_row]).to_csv(
                    partial_file,
                    mode='a',
                    header=not os.path.exists(partial_file),
                    index=False
                )
                print(f"Saved progress: {image_name}")
            except Exception as e:
                print(f"Error saving progress: {str(e)}")

    # Merge and save all results finally
    try:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_csv, index=False)

        if os.path.exists(partial_file):
            os.remove(partial_file)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        print(f"\n✅ Final results saved to {output_csv}")
    except Exception as e:
        print(f"\n❌ Error saving final results: {str(e)}")
        print("Partial results remain in file.")

    return results

# 主函数
def main():
    CONFIG = {
        # DATA JSON file path 
        "json_path":r"./data/multi-hop-data/Zero-hop.json",
        # Image directory path
        "image_dir": r"./Image/SD3-Zero-hop",
        # OpenAI API key
        "api_key": "sk-***********************",
        "output_csv": "GPT4o-DALL-E3-Two-hop.csv",
        "request_delay": 0.5
    }

    prompts_data = load_prompts(CONFIG['json_path'])
    if not prompts_data:
        print("Error: Failed to load prompt data")
        return

    process_images_and_questions(
        prompts_data=prompts_data,
        image_dir=CONFIG['image_dir'],
        api_key=CONFIG['api_key'],
        output_csv=CONFIG['output_csv'],
        delay=CONFIG['request_delay']
    )

if __name__ == "__main__":
    main()
