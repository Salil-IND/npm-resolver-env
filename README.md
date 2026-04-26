# 🧠 AutoResolve: RL-Driven Dependency Manager
**Meta OpenEnv Hackathon 2026 Submission**

AutoResolve is an advanced Reinforcement Learning (RL) environment built on **OpenEnv** that teaches an LLM how to resolve complex Node.js `peerDependency` conflicts using Generative Reward Policy Optimization (GRPO). 

🔗 **[Hugging Face Space (Environment)](https://huggingface.co/spaces/ArpitBaliyan/npm-resolver)** *(<- Insert your HF Space link here)* 🔗 **[Interactive Training & UI Notebook (Colab)](#)** *(<- Insert your Colab share link here)* ## 1. The Problem: Dependency Hell
"Dependency hell" is a universal developer bottleneck. Standard code-generation models (like Claude or Llama) struggle to fix `package.json` conflicts because they rely on pattern matching and often hallucinate package versions. Semantic Versioning (SemVer) is a strict mathematical constraint, not a creative writing exercise. They lack the ability to "test, fail, and adapt."

## 2. The Environment (OpenEnv)
We built a strict, Gym-style environment (`env.py` and `openenv.yaml`) that acts as a simulated NPM registry.

* **The State:** The agent is fed a broken `package.json` and a terminal error log (e.g., `ERESOLVE`).
* **The Action:** The agent must output a strict JSON action: `{"package_to_update": "react", "new_version": "^18.0.0"}`.
* **The Reward Function:** * `+50` for successfully resolving the dependency tree.
  * `+10` for making progress (fixing one conflict but exposing another).
  * `-5` for invalid JSON formatting or missing the caret (`^`) symbol.
  * `-100` for hallucinating packages or destroying core dependencies (Anti-Cheat).

## 3. Training & Architecture
We trained **Llama-3 (8B)** using **4-bit quantization via Unsloth** and **Hugging Face TRL's GRPOTrainer**. Training was conducted entirely on a single Google Colab T4 GPU, proving that highly effective RL environments can be run without massive compute clusters.

## 4. Rigorous Evaluation & Results
To prove our RL agent learned the underlying math of SemVer rather than just memorizing training data, we used an independent LLM (DeepSeek) to act as a QA Engineer. It generated 15 blind, cross-ecosystem conflicts (React, Vue, Express, Mongoose, Webpack).

**Our agent scored a 93.3% Zero-Shot Accuracy (70/75)**. It flawlessly resolved deep peer-dependency missing links and version downgrades. 

## 5. How to Run Locally
1. Open our [Google Colab Notebook](#).
2. Run the initialization cells to download the trained weights from Hugging Face.
3. Run the final cell to launch the **Gradio UI** right in your browser.
4. Use the "1-Click Test Scenarios" to watch the RL Agent resolve conflicts in real-time.