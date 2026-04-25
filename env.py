import json
import random
from dataclasses import dataclass

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
        self.max_steps = 10
        self.current_state = {}
        self.step_count = 0
        self.last_action = None
        
        # A massive vocabulary of real and fake packages to force the AI to learn grammar, not names.
        self.vocab = [
            "react", "express", "lodash", "mongoose", "axios", "webpack", 
            "jest", "chalk", "next", "tailwindcss", "typescript", "eslint",
            "alpha-lib", "beta-core", "gamma-utils", "delta-auth", "omega-router"
        ]
        
        self.active_rule = {}

    def _generate_error_log(self, dependencies) -> str:
        """Evaluates the current state against the randomly generated rule."""
        if not self.active_rule:
            return "SUCCESS: Audited packages in 0.01s. No vulnerabilities found."
            
        root = self.active_rule["root"]
        peer = self.active_rule["peer"]
        req_ver = self.active_rule["req_peer_ver"]
        
        # If the root package is gone, the conflict disappears
        if root not in dependencies:
            return "SUCCESS: Audited packages in 0.01s. No vulnerabilities found."

        if peer not in dependencies:
            return f"Missing peer dependency: {root}@{self.active_rule['root_ver']} requires {peer}@{req_ver}"
        elif dependencies[dependencies.get(peer, "")] != req_ver and peer in dependencies:
            return f"Conflict: {root}@{self.active_rule['root_ver']} requires {peer}@{req_ver}, but found {dependencies[peer]}"
            
        return "SUCCESS: Audited packages in 0.01s. No vulnerabilities found."

    def reset(self) -> Observation:
        """Generates a completely randomized dependency conflict every single time."""
        self.step_count = 0
        self.last_action = None
        
        # 1. Pick 3 random packages
        pkgs = random.sample(self.vocab, 3)
        root_pkg, peer_pkg, extra_pkg = pkgs
        
        # 2. Generate random semantic versions
        bad_version = f"^{random.randint(1, 5)}.{random.randint(0, 5)}.0"
        req_version = f"^{random.randint(6, 12)}.{random.randint(0, 5)}.0"
        root_version = f"{random.randint(1, 4)}.0.0"
        
        # 3. Set the active broken state
        self.current_state = {
            root_pkg: root_version,
            peer_pkg: bad_version,
            extra_pkg: "^1.0.0"
        }
        
        # 4. Define the hidden rule the LLM must discover
        self.active_rule = {
            "root": root_pkg,
            "root_ver": root_version,
            "peer": peer_pkg,
            "req_peer_ver": req_version
        }
        
        # 5. Randomly decide if it's a "Conflict" or a "Missing" error
        if random.random() > 0.5:
            del self.current_state[peer_pkg] # Simulate missing dependency
            
        initial_errors = self._generate_error_log(self.current_state)
        
        return Observation(
            current_package_json=json.dumps(self.current_state, indent=2),
            npm_error_log=initial_errors,
            step_count=self.step_count
        )

    def step(self, action: Action):
        self.step_count += 1
        reward = 0
        done = False
        info = {}

        if self.step_count >= self.max_steps:
            obs = Observation(json.dumps(self.current_state, indent=2), "TIMEOUT", self.step_count)
            return obs, -10, True, info

        pkg = action.package_to_update
        new_ver = action.new_version

        if (pkg, new_ver) == self.last_action:
            reward -= 2
        self.last_action = (pkg, new_ver)

        if not isinstance(new_ver, str) or (not new_ver.startswith("^") and new_ver != "DELETE" and "." not in new_ver):
            obs = Observation(json.dumps(self.current_state), "Invalid version format.", self.step_count)
            return obs, -5, False, info

        # Apply Action
        if new_ver == "DELETE":
            self.current_state.pop(pkg, None)
        else:
            self.current_state[pkg] = new_ver

        if len(self.current_state) < 1:
            obs = Observation(json.dumps(self.current_state), "FATAL: Empty state.", self.step_count)
            return obs, -100, True, info

        reward -= 1 

        error_log = self._generate_error_log(self.current_state)
        
        if "SUCCESS" in error_log:
            reward += 50
            done = True

        obs = Observation(
            current_package_json=json.dumps(self.current_state, indent=2),
            npm_error_log=error_log,
            step_count=self.step_count
        )
        
        return obs, reward, done, info