from dataclasses import dataclass
import json

@dataclass
class Action:
    package_to_update: str
    new_version: str

@dataclass
class Observation:
    current_package_json: str
    npm_error_log: str
    step_count: int




class NPMResolverEnv:
    def __init__(self):
        # THE MOCK REGISTRY: The absolute rules of the universe.
        self.registry = {
            "react-router-dom": {
                "6.0.0": {"requires": {"react": "^18.0.0", "react-dom": "^18.0.0"}}
            },
            "framer-motion": {
                "10.0.0": {"requires": {"react": "^18.0.0"}}
            }
        }
        # ANTI-CHEAT: Packages the agent is NEVER allowed to delete
        self.untouchable_packages = ["react", "react-dom"]
        self.max_steps = 10
        self.current_state = {}
        self.step_count = 0

    def _generate_error_log(self, dependencies) -> str:
        """The Verifier: Checks current dependencies against the registry."""
        errors = []
        for pkg, version in dependencies.items():
            if pkg in self.registry and version in self.registry[pkg]:
                requirements = self.registry[pkg][version]["requires"]
                for req_pkg, req_ver in requirements.items():
                    if req_pkg not in dependencies:
                        errors.append(f"Missing peer dependency: {pkg}@{version} requires {req_pkg}@{req_ver}")
                    elif dependencies[req_pkg] != req_ver:
                        errors.append(f"Conflict: {pkg}@{version} requires {req_pkg}@{req_ver}, but found {dependencies[req_pkg]}")
        
        return "\n".join(errors) if errors else "SUCCESS: Audited packages in 0.01s. No vulnerabilities found."

    def reset(self) -> Observation:
        """Starts a new episode with a broken package.json."""
        self.step_count = 0
        
        # This is Level 1 Difficulty: React is outdated, causing router/framer to fail.
        self.current_state = {
            "react": "^17.0.0",
            "react-dom": "^17.0.0",
            "react-router-dom": "6.0.0",
            "framer-motion": "10.0.0"
        }
        
        initial_errors = self._generate_error_log(self.current_state)
        
        return Observation(
            current_package_json=json.dumps(self.current_state, indent=2),
            npm_error_log=initial_errors,
            step_count=self.step_count
        )

    def step(self, action: Action):
        """Receives the LLM's action, updates state, and calculates reward."""
        self.step_count += 1
        reward = 0
        done = False

        # 1. Apply the Action
        pkg = action.package_to_update
        new_ver = action.new_version

        # 2. Check for Anti-Cheat (Did it try to delete React to bypass errors?)
        if new_ver == "DELETE":
            if pkg in self.untouchable_packages:
                reward = -100  # The Nuke Penalty
                done = True
                obs = Observation(json.dumps(self.current_state), "FATAL: Cannot remove core framework.", self.step_count)
                return obs, reward, done
            else:
                self.current_state.pop(pkg, None)
        else:
            self.current_state[pkg] = new_ver

        # 3. Efficiency Penalty (Encourage fast fixes)
        reward -= 1 

        # 4. Verify the new state
        error_log = self._generate_error_log(self.current_state)
        
        if "SUCCESS" in error_log:
            reward += 50  # Ultimate Success!
            done = True
        elif self.step_count >= self.max_steps:
            reward -= 10  # Timeout
            done = True

        obs = Observation(
            current_package_json=json.dumps(self.current_state, indent=2),
            npm_error_log=error_log,
            step_count=self.step_count
        )
        
        return obs, reward, done
















if __name__ == "__main__":
    print("--- STARTING LOCAL ENVIRONMENT TEST ---")
    env = NPMResolverEnv()
    
    # 1. Start the Environment
    obs = env.reset()
    print(f"\nINITIAL STATE:\n{obs.current_package_json}")
    print(f"\nINITIAL ERRORS:\n{obs.npm_error_log}")
    
    # 2. Simulate a BAD AI Action (Fails to fix it)
    print("\n--- AI TAKES BAD ACTION ---")
    bad_action = Action(package_to_update="framer-motion", new_version="9.0.0")
    obs, reward, done = env.step(bad_action)
    print(f"Reward: {reward}")
    
    # 3. Simulate a CHEATING AI Action (Tries to delete React)
    print("\n--- AI TRIES TO CHEAT ---")
    cheat_action = Action(package_to_update="react", new_version="DELETE")
    obs, reward, done = env.step(cheat_action)
    print(f"Reward: {reward} (Expect -100)")
    print(f"Error: {obs.npm_error_log}")
    
    # 4. Simulate a GOOD AI Action (Fixes the problem)
    env.reset() # Reset world
    print("\n--- AI TAKES GOOD ACTION ---")
    good_action_1 = Action(package_to_update="react", new_version="^18.0.0")
    env.step(good_action_1)
    good_action_2 = Action(package_to_update="react-dom", new_version="^18.0.0")
    obs, reward, done = env.step(good_action_2)
    
    print(f"Final Reward: {reward} (Expect positive)")
    print(f"Final Log: {obs.npm_error_log}")