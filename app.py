from flask import Flask, render_template, request, send_file, jsonify
import json
import os
import re
import requests
from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv
from collections import deque

app = Flask(__name__)
load_dotenv()

OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep only the last N runs to avoid filling disk
MAX_FILES = 30
def cleanup_old_files():
    files = sorted(
        [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)],
        key=os.path.getmtime
    )
    if len(files) > MAX_FILES:
        for old in files[:-MAX_FILES]:
            try:
                os.remove(old)
            except Exception:
                pass

def extract_json(content):
    """Extract the first JSON object from raw text safely."""
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    return match.group(1).strip() if match else content.strip()

def clean_json_file(file_path):
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return f"JSON file {file_path} is missing or empty"

        with open(file_path, 'r') as f:
            content = f.read().strip()

        if not content:
            return f"JSON file {file_path} is empty after stripping"

        cleaned_content = extract_json(content)

        try:
            json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            return f"Invalid JSON in {file_path} after cleaning: {str(e)}"

        with open(file_path, 'w') as f:
            f.write(cleaned_content)
        return None
    except Exception as e:
        return str(e)

gemini_key = os.getenv('GOOGLE_API_KEY')
if not gemini_key:
    raise ValueError("GOOGLE_API_KEY is required")
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_key,
    temperature=0.7
)

mermaid_generator = Agent(
    role='Mermaid Code Generator',
    goal='Generate three distinct Mermaid flowchart interpretations with unique logical structures based on a user-provided description.',
    backstory='You are an expert in diagramming, skilled in creating valid and diverse Mermaid syntax for flowcharts.',
    verbose=True,
    llm=gemini_llm,
    tools=[]
)

flowchart_renderer = Agent(
    role='Flowchart Renderer',
    goal='Render Mermaid flowchart code into SVG and PNG images using the Kroki API.',
    backstory='You are a technical illustrator with expertise in rendering high-quality diagrams.',
    verbose=True,
    llm=gemini_llm,
    tools=[]
)

def validate_mermaid_code(code: str) -> bool:
    """Validate Mermaid code more flexibly."""
    if not code or not isinstance(code, str):
        return False

    # Must start with 'graph' and direction
    if not re.search(r'graph\s+(TD|LR|BT|RL)', code, re.IGNORECASE):
        return False

    # Must contain at least one connection (--> or ---)
    if not any(link in code for link in ['-->', '---']):
        return False

    # Accept either [ ] style nodes OR subgraph definitions OR plain nodes
    if not re.search(r'\[.*?\]', code) and "subgraph" not in code.lower() and re.search(r'[A-Za-z0-9]+\s*-->', code) is None:
        return False

    return True


def render_with_kroki(mermaid_code, variant, fmt):
    """Render Mermaid code via Kroki API into SVG or PNG."""
    url = f"https://kroki.io/mermaid/{fmt}"
    response = requests.post(url, data=mermaid_code.encode("utf-8"))
    if response.status_code != 200:
        raise ValueError(f"Kroki rendering failed ({fmt}): {response.text}")

    path = os.path.join(OUTPUT_DIR, variant[f"{fmt}_file"])
    mode = "wb" if fmt == "png" else "w"
    with open(path, mode) as f:
        if fmt == "png":
            f.write(response.content)
        else:
            f.write(response.text)
    return f"static/outputs/{variant[f'{fmt}_file']}"

def run_crew(flowchart_description):
    cleanup_old_files()

    variants = [
        {'id': 1, 'name': 'Variant 1', 'json_file': 'mermaid_code_variant1.json', 'svg_file': 'flowchart_output_variant1.svg', 'png_file': 'flowchart_output_variant1.png'},
        {'id': 2, 'name': 'Variant 2', 'json_file': 'mermaid_code_variant2.json', 'svg_file': 'flowchart_output_variant2.svg', 'png_file': 'flowchart_output_variant2.png'},
        {'id': 3, 'name': 'Variant 3', 'json_file': 'mermaid_code_variant3.json', 'svg_file': 'flowchart_output_variant3.svg', 'png_file': 'flowchart_output_variant3.png'}
    ]

    result_json = {'variants': []}

    for variant in variants:
        generate_mermaid_task = Task(
            description=(
                f"Based on the user-provided flowchart description, generate a unique Mermaid flowchart in JSON format:\n"
                f"Description: {flowchart_description}\n"
                f"This is variant {variant['id']} of 3. Create a distinct logical structure (linear, branching, or parallel).\n"
                f"Output strictly as a JSON object like: {{\"mermaid_code\": \"graph TD; A[Start] --> B[Process] --> C[End]\"}}"
            ),
            expected_output="A JSON object with valid Mermaid flowchart code under 'mermaid_code'.",
            agent=mermaid_generator,
            output_file=os.path.join(OUTPUT_DIR, variant['json_file'])
        )

        crew = Crew(
            agents=[mermaid_generator],
            tasks=[generate_mermaid_task],
            verbose=True
        )

        try:
            crew.kickoff()
            json_file = os.path.join(OUTPUT_DIR, variant['json_file'])
            if not os.path.exists(json_file):
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"JSON file missing: {json_file}"})
                continue

            json_error = clean_json_file(json_file)
            if json_error:
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"Failed to clean JSON file: {json_error}"})
                continue

            with open(json_file, 'r') as f:
                mermaid_data = json.load(f)

            mermaid_code = mermaid_data.get('mermaid_code', '')
            if not validate_mermaid_code(mermaid_code):
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': 'Invalid or empty Mermaid code'})
                continue

            variant_result = {'id': variant['id'], 'name': variant['name']}

            try:
                variant_result['svg_path'] = render_with_kroki(mermaid_code, variant, "svg")
                variant_result['png_path'] = render_with_kroki(mermaid_code, variant, "png")

                result_file = os.path.join(OUTPUT_DIR, f"flowchart_result_variant{variant['id']}.json")
                with open(result_file, 'w') as f:
                    json.dump(variant_result, f, indent=2)

                result_json['variants'].append(variant_result)
            except Exception as e:
                variant_result['error'] = f"Rendering failed: {str(e)}"
                result_json['variants'].append(variant_result)

        except Exception as e:
            result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"Crew execution error: {str(e)}"})

    return result_json

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        flowchart_description = request.form.get('description')
        if not flowchart_description:
            error = "Please provide a flowchart description."
        else:
            result = run_crew(flowchart_description)
            if not result.get('variants'):
                error = "No flowchart variants were generated."
            elif all('error' in v for v in result['variants']):
                error = "All variants failed: " + "; ".join(v['error'] for v in result['variants'])
                result = None
            elif any('error' in v for v in result['variants']):
                error = "Some variants failed. See details below."

    return render_template('index.html', result=result or {}, error=error)

@app.route('/download/<file_type>/<variant_id>')
def download(file_type, variant_id):
    variant_files = {
        '1': {'svg': 'flowchart_output_variant1.svg', 'png': 'flowchart_output_variant1.png'},
        '2': {'svg': 'flowchart_output_variant2.svg', 'png': 'flowchart_output_variant2.png'},
        '3': {'svg': 'flowchart_output_variant3.svg', 'png': 'flowchart_output_variant3.png'}
    }
    if variant_id not in variant_files or file_type not in ['svg', 'png']:
        return jsonify({'error': 'Invalid file type or variant ID'}), 400

    file_path = os.path.join(OUTPUT_DIR, variant_files[variant_id][file_type])
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return jsonify({'error': f'{file_type.upper()} file missing or empty for variant {variant_id}'}), 404

    return send_file(file_path, as_attachment=True, download_name=variant_files[variant_id][file_type])

if __name__ == '__main__':
    app.run(debug=True)
