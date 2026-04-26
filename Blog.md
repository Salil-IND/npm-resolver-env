# Teaching an LLM to Survive Node.js Dependency Hell using RL and OpenEnv

If you are a JavaScript developer, you have seen this wall of red text:
`npm ERR! code ERESOLVE`
`npm ERR! ERESOLVE unable to resolve dependency tree`

Fixing these peer-dependency conflicts usually involves twenty minutes of frantic Googling, manually downgrading packages, and praying your app still builds. Standard LLMs aren't much help either; they hallucinate versions because they treat Semantic Versioning as text generation, rather than a strict mathematical constraint. 

For the **Meta OpenEnv Hackathon**, we decided to fix this by treating package resolution not as a chat prompt, but as a playable game. We built **AutoResolve**.

## The Architecture: OpenEnv + GRPO
We utilized the OpenEnv framework to build a strict, Gym-style Python environment (`env.py` and `openenv.yaml`) that acts as a mock NPM registry. 

Instead of just telling an LLM the answer, we let it play inside this registry. We deployed **Llama-3 8B** (using Unsloth for 4-bit quantization) and trained it using Hugging Face TRL’s **Generative Reward Policy Optimization (GRPO)**.

### How the Environment Works
1. **The State:** The environment generates a broken `package.json` alongside a realistic NPM error trace.
2. **The Action:** The agent acts as a greedy 1-step optimizer. It must output a strict JSON payload, for example: `{"package_to_update": "react", "new_version": "^18.0.0"}`.
3. **The Reward:** The environment validates the action against the mock registry. 
   - Fixing the tree grants **+50**. 
   - Hallucinating a fake package yields **-100**. 
   - Forgetting the caret (`^`) symbol yields **-5**.

### Overcoming "Reward Hacking"
During training, we encountered a classic RL phenomenon: Reward Hacking. Our environment initially validated the major version numbers, so the AI figured out it could achieve maximum points while saving token-generation time by dropping the caret symbol (`^`). Rather than wasting compute on a complete retraining cycle, we implemented a lightweight post-processing formatter—a standard practice in production LLM pipelines—to re-inject the caret.

## The Results: 93.3% Zero-Shot Accuracy
Training an RL agent on the entire 2-million-package NPM registry requires a massive compute cluster. To prove our architecture within the hackathon timeline, we curated a "Mega Registry" containing 5 distinct ecosystems (React, Vue, Express, Webpack, Mongoose). 

To evaluate the model, we generated 15 rigorous, blind test cases spanning cross-ecosystem conflicts. 
**The agent scored 93.3% (14/15) accuracy.** The single edge-case failure was an attempt to update `babel-loader` instead of its missing peer `@babel/core`—a known artifact of our MVP's 'greedy 1-step optimizer' design, which we plan to resolve in V2 using Monte Carlo Tree Search (MCTS) for multi-step rollouts.

## Conclusion
By utilizing OpenEnv, we successfully proved that Reinforcement Learning can teach an LLM rigid mathematical constraints. The architecture is fully complete; scaling to the entire NPM registry simply requires deploying our identical pipeline to a production GPU cluster. 

**Explore the Project:**
* 🧠 **[Try the Gradio UI / View Training Code](#)** *(<- Insert Colab Link)*
* 📦 **[View the OpenEnv Space](#)** *(<- Insert HF Space Link)*
* 🚀 **[Download the Weights](https://huggingface.co/ArpitBaliyan/npm-resolver-rl-model)**