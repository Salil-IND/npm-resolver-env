import gradio as gr
import json
import torch
from unsloth import FastLanguageModel

# 1. Load from the CLOUD (Ensures it works on HF Spaces)
print("Loading trained model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ArpitBaliyan/npm-resolver-rl-model", # 👈 Change this to your actual HF repo name
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def resolve_npm_conflict(pkg_json_str, error_log_str):
    system_prompt = """You are an expert Node.js dependency resolver.
You must output a SINGLE JSON object with keys: "package_to_update" and "new_version".
Do not output markdown or explanations. Just raw JSON.
CRITICAL RULE: You MUST include the caret symbol (^) in the new_version (e.g., "^18.0.0")."""

    user_prompt = f"PACKAGE.JSON:\n{pkg_json_str}\n\nERRORS:\n{error_log_str}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
    generated_tokens = outputs[0][inputs.shape[1]:]
    ai_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    try:
        # Clean the response in case the AI added markdown backticks
        clean_json = ai_response.replace("```json", "").replace("```", "").strip()
        action = json.loads(clean_json)
        
        pkg = action.get("package_to_update")
        ver = action.get("new_version")

        current_data = json.loads(pkg_json_str)
        
        # Check if "dependencies" key exists, otherwise use root
        deps = current_data.get("dependencies", current_data)
        
        if ver == "DELETE":
            deps.pop(pkg, None)
        else:
            deps[pkg] = ver

        # If we had to go into a sub-key, put it back
        if "dependencies" in current_data:
            current_data["dependencies"] = deps
        else:
            current_data = deps

        return clean_json, json.dumps(current_data, indent=2)
    except Exception as e:
        return ai_response, f"Error applying fix: {str(e)}"

# UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 AutoResolve RL: AI Dependency Fixer")
    
    with gr.Row():
        with gr.Column():
            in_pkg = gr.Code(label="Current package.json", language="json", lines=10)
            in_err = gr.Textbox(label="NPM Error Log", lines=4)
            btn = gr.Button("🔮 Resolve Dependencies", variant="primary")

        with gr.Column():
            out_action = gr.Code(label="AI Action", language="json")
            out_pkg = gr.Code(label="Fixed package.json", language="json")

    # Examples for the judges to click
    gr.Examples(
        examples=[
            ['{"dependencies": {"react": "^17.0.0", "react-router-dom": "6.0.0"}}', "Conflict: react-router-dom@6.0.0 requires react@^18.0.0"],
        ],
        inputs=[in_pkg, in_err]
    )

    btn.click(resolve_npm_conflict, inputs=[in_pkg, in_err], outputs=[out_action, out_pkg])

demo.launch()