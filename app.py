from flask import Flask, render_template, request, send_file, jsonify
import json
import logging
import os
import re
import subprocess
from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.getcwd(), 'static', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to clean JSON file by removing ```json and ``` markers and preceding text
def clean_json_file(file_path):
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.error(f"JSON file {file_path} is missing or empty")
            return f"JSON file {file_path} is missing or empty"
        
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            logger.error(f"JSON file {file_path} is empty after stripping")
            return f"JSON file {file_path} is empty after stripping"
        
        # Log raw content for debugging
        logger.debug(f"Raw content of {file_path}: {content}")
        
        # Extract JSON content between ```json and ``` markers, or take the entire content if no markers
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            cleaned_content = json_match.group(1).strip()
        else:
            # If no markers, assume the content is meant to be JSON
            cleaned_content = content.strip()
        
        # Check if cleaned content resembles JSON
        if not (cleaned_content.startswith('{') and cleaned_content.endswith('}')):
            logger.error(f"Cleaned content of {file_path} does not resemble JSON: {cleaned_content}")
            return f"Cleaned content of {file_path} does not resemble JSON"
        
        # Validate JSON
        try:
            json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path} after cleaning: {str(e)}")
            return f"Invalid JSON in {file_path} after cleaning: {str(e)}"
        
        # Write cleaned content back
        with open(file_path, 'w') as f:
            f.write(cleaned_content)
        logger.debug(f"Cleaned JSON file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error cleaning JSON file {file_path}: {str(e)}")
        return str(e)

# Initialize Gemini LLM
gemini_key = os.getenv('GOOGLE_API_KEY')
if not gemini_key:
    logger.error("GOOGLE_API_KEY not found in .env file")
    raise ValueError("GOOGLE_API_KEY is required")
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_key,
    temperature=0.7
)

# Agent 1: Mermaid Code Generator
mermaid_generator = Agent(
    role='Mermaid Code Generator',
    goal='Generate three distinct Mermaid flowchart interpretations with unique logical structures based on a user-provided description.',
    backstory='You are an expert in diagramming, skilled in creating valid and diverse Mermaid syntax for flowcharts.',
    verbose=True,
    llm=gemini_llm,
    tools=[]
)

# Agent 2: Flowchart Renderer
flowchart_renderer = Agent(
    role='Flowchart Renderer',
    goal='Render Mermaid flowchart code into SVG and PNG images using the Kroki API via curl commands.',
    backstory='You are a technical illustrator with expertise in API integrations and rendering high-quality diagrams using Kroki.',
    verbose=True,
    llm=gemini_llm,
    tools=[]
)

def validate_mermaid_code(code):
    """Validate Mermaid code for basic syntax requirements."""
    if not code or not isinstance(code, str):
        logger.error("Mermaid code is empty or not a string")
        return False
    # Check for graph definition and at least one node connection
    if 'graph' not in code.lower() or '-->' not in code:
        logger.error("Mermaid code missing graph definition or connections")
        return False
    # Basic syntax check for nodes (e.g., A[Label])
    if not re.search(r'[A-Za-z0-9]+\[.*?\]', code):
        logger.error("Mermaid code missing valid node definitions")
        return False
    return True

def run_crew(flowchart_description):
    variants = [
        {'id': 1, 'name': 'Variant 1', 'json_file': 'mermaid_code_variant1.json', 'svg_file': 'flowchart_output_variant1.svg', 'png_file': 'flowchart_output_variant1.png'},
        {'id': 2, 'name': 'Variant 2', 'json_file': 'mermaid_code_variant2.json', 'svg_file': 'flowchart_output_variant2.svg', 'png_file': 'flowchart_output_variant2.png'},
        {'id': 3, 'name': 'Variant 3', 'json_file': 'mermaid_code_variant3.json', 'svg_file': 'flowchart_output_variant3.svg', 'png_file': 'flowchart_output_variant3.png'}
    ]

    result_json = {'variants': []}

    for variant in variants:
        logger.debug(f"Processing variant {variant['id']} ({variant['name']})")
        # Task 1: Generate Mermaid Code for Variant
        generate_mermaid_task = Task(
            description=(
                f"Based on the user-provided flowchart description, generate a unique Mermaid flowchart in JSON format:\n"
                f"Description: {flowchart_description}\n"
                f"This is variant {variant['id']} of 3. Create a distinct logical structure (e.g., linear, decision-based, or parallel processes) that differs from other variants but aligns with the description’s intent. "
                f"Use simple Mermaid syntax (e.g., graph TD; A[Start] --> B[Process] --> C[End]). Include minimal styling (%%{{init: {{'theme': 'base'}}}}%%). "
                f"Ensure the code is valid and renderable by Kroki. Output **only** a pure JSON object with a single key 'mermaid_code' containing the Mermaid code as a string, e.g., {{'mermaid_code': 'graph TD; A[Start] --> B[Process] --> C[End]'}}. "
                f"Do **not** include ```json or ``` markers, comments, explanatory text, or any other content outside the JSON object. Any additional text will cause errors."
            ),
            expected_output="A JSON object with valid Mermaid flowchart code under 'mermaid_code'.",
            agent=mermaid_generator,
            output_file=os.path.join(OUTPUT_DIR, variant['json_file'])
        )

        # Task 2: Render Flowchart for Variant
        render_flowchart_task = Task(
            description=(
                f"Using the Mermaid code from {variant['json_file']}, render it as SVG and PNG images using the Kroki API via curl commands. "
                f"Send the code to 'https://kroki.io/mermaid/svg' and 'https://kroki.io/mermaid/png' with --data-raw. "
                f"Save the SVG to '{variant['svg_file']}' and PNG to '{variant['png_file']}' in static/outputs. "
                f"Return a JSON object with the file paths or an error message if rendering fails."
            ),
            expected_output=f"A JSON object with paths to '{variant['svg_file']}' and '{variant['png_file']}' or an error message.",
            agent=flowchart_renderer,
            output_file=os.path.join(OUTPUT_DIR, f"flowchart_result_variant{variant['id']}.json"),
            context=[generate_mermaid_task]
        )

        # Initialize Crew for Variant
        crew = Crew(
            agents=[mermaid_generator, flowchart_renderer],
            tasks=[generate_mermaid_task, render_flowchart_task],
            verbose=True
        )

        # Execute Crew
        try:
            crew.kickoff()
            # Log raw JSON file content before cleaning
            json_file = os.path.join(OUTPUT_DIR, variant['json_file'])
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    raw_content = f.read()
                logger.debug(f"Raw JSON content for variant {variant['id']}: {raw_content}")
            else:
                logger.error(f"JSON file missing for variant {variant['id']}: {json_file}")
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"JSON file missing: {json_file}"})
                continue
            
            # Clean the JSON file
            json_error = clean_json_file(json_file)
            if json_error:
                logger.error(f"Variant {variant['id']} JSON cleaning failed: {json_error}")
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"Failed to clean JSON file: {json_error}"})
                continue
            
            # Read and validate Mermaid code
            try:
                with open(json_file, 'r') as f:
                    mermaid_data = json.load(f)
                mermaid_code = mermaid_data.get('mermaid_code', '')
                logger.debug(f"Variant {variant['id']} Mermaid code: {mermaid_code}")
                if not validate_mermaid_code(mermaid_code):
                    logger.error(f"Variant {variant['id']} has invalid Mermaid code")
                    result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': 'Invalid or empty Mermaid code generated'})
                    continue
            except Exception as e:
                logger.error(f"Variant {variant['id']} Mermaid code read failed: {str(e)}")
                result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': f"Failed to read Mermaid code: {str(e)}"})
                continue

            # Execute curl commands to render SVG and PNG
            variant_result = {'id': variant['id'], 'name': variant['name']}
            try:
                # Render SVG
                curl_svg_command = ['curl', 'https://kroki.io/mermaid/svg', '--data-raw', mermaid_code]
                svg_path = os.path.join(OUTPUT_DIR, variant['svg_file'])
                with open(svg_path, 'w') as svg_file:
                    result = subprocess.run(curl_svg_command, capture_output=True, text=True, check=True)
                    svg_file.write(result.stdout)
                variant_result['svg_path'] = f"static/outputs/{variant['svg_file']}"
                logger.debug(f"Generated SVG for variant {variant['id']}: {svg_path}")
                
                # Render PNG
                curl_png_command = ['curl', 'https://kroki.io/mermaid/png', '--data-raw', mermaid_code]
                png_path = os.path.join(OUTPUT_DIR, variant['png_file'])
                with open(png_path, 'wb') as png_file:
                    result = subprocess.run(curl_png_command, capture_output=True, check=True)
                    png_file.write(result.stdout)
                variant_result['png_path'] = f"static/outputs/{variant['png_file']}"
                logger.debug(f"Generated PNG for variant {variant['id']}: {png_path}")
                
                # Verify files exist and are non-empty
                if not os.path.exists(svg_path) or os.path.getsize(svg_path) == 0:
                    variant_result['error'] = 'SVG file was not created or is empty'
                    logger.error(f"SVG file invalid for variant {variant['id']}: {svg_path}")
                elif not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
                    variant_result['error'] = 'PNG file was not created or is empty'
                    logger.error(f"PNG file invalid for variant {variant['id']}: {png_path}")
                
                # Save variant JSON result
                result_file = os.path.join(OUTPUT_DIR, f"flowchart_result_variant{variant['id']}.json")
                with open(result_file, 'w') as f:
                    json.dump(variant_result, f, indent=2)
                logger.debug(f"Saved result for variant {variant['id']}: {result_file}")
                
                result_json['variants'].append(variant_result)
                logger.debug(f"Added variant {variant['id']} to result_json: {variant_result}")
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to render images for variant {variant['id']}: {e.stderr}"
                variant_result['error'] = error_msg
                result_json['variants'].append(variant_result)
                logger.error(error_msg)
        except Exception as e:
            error_msg = f"Error during crew execution for variant {variant['id']}: {str(e)}"
            result_json['variants'].append({'id': variant['id'], 'name': variant['name'], 'error': error_msg})
            logger.error(error_msg)
    
    logger.debug(f"Final result_json: {json.dumps(result_json, indent=2)}")
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
                error = "No flowchart variants were generated. Check server logs for details."
            elif all('error' in variant for variant in result['variants']):
                error = "Errors occurred in all variants: " + "; ".join(v['error'] for v in result['variants'] if v.get('error'))
                result = None
            elif any('error' in variant for variant in result['variants']):
                error = "Some variants failed to generate. Check the details below."
    
    return render_template('index.html', result=result, error=error)

@app.route('/download/<file_type>/<variant_id>')
def download(file_type, variant_id):
    variant_files = {
        '1': {'svg': 'flowchart_output_variant1.svg', 'png': 'flowchart_output_variant1.png'},
        '2': {'svg': 'flowchart_output_variant2.svg', 'png': 'flowchart_output_variant2.png'},
        '3': {'svg': 'flowchart_output_variant3.svg', 'png': 'flowchart_output_variant3.png'}
    }
    if variant_id not in variant_files or file_type not in ['svg', 'png']:
        logger.error(f"Invalid file type '{file_type}' or variant ID '{variant_id}'")
        return jsonify({'error': 'Invalid file type or variant ID'}), 400
    
    file_path = os.path.join(OUTPUT_DIR, variant_files[variant_id][file_type])
    
    # Check if file exists and is non-empty
    if not os.path.exists(file_path):
        logger.error(f"Download failed: {file_path} not found")
        return jsonify({'error': f'{file_type.upper()} file not found for variant {variant_id}'}), 404
    
    if os.path.getsize(file_path) == 0:
        logger.error(f"Download failed: {file_path} is empty")
        return jsonify({'error': f'{file_type.upper()} file is empty for variant {variant_id}'}), 404
    
    # Validate file type
    try:
        with open(file_path, 'rb') as f:
            header = f.read(50)  # Read first 50 bytes for validation
            if file_type == 'png':
                if header[:8] != b'\x89PNG\r\n\x1a\n':
                    logger.error(f"Download failed: {file_path} is not a valid PNG file")
                    return jsonify({'error': f'Invalid PNG file for variant {variant_id}'}), 404
            elif file_type == 'svg':
                header_str = header.decode('utf-8', errors='ignore').lower()
                if not ('<?xml' in header_str or '<svg' in header_str):
                    logger.error(f"Download failed: {file_path} is not a valid SVG file")
                    return jsonify({'error': f'Invalid SVG file for variant {variant_id}'}), 404
    except Exception as e:
        logger.error(f"Download failed: Error reading {file_path}: {str(e)}")
        return jsonify({'error': f'Error validating {file_type.upper()} file for variant {variant_id}'}), 404
    
    try:
        return send_file(file_path, as_attachment=True, download_name=variant_files[variant_id][file_type])
    except Exception as e:
        logger.error(f"Download failed: Error sending {file_path}: {str(e)}")
        return jsonify({'error': f'Failed to download {file_type.upper()} file for variant {variant_id}'}), 500

if __name__ == '__main__':
    app.run(debug=True)