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
        # 1. The Rules of the Universe
        self.registry = {
            "react-router-dom": {
                "6.0.0": {"requires": {"react": "^18.0.0", "react-dom": "^18.0.0"}},
                "5.0.0": {"requires": {"react": "^16.8.0", "react-dom": "^16.8.0"}}
            },
            "framer-motion": {
                "10.0.0": {"requires": {"react": "^18.0.0"}},
                "4.0.0": {"requires": {"react": "^17.0.0"}}
            }
        }
        
        # 2. Scenarios (Fix #7: Randomization)
        self.scenarios = [
            {
                "react": "^17.0.0",
                "react-dom": "^17.0.0",
                "react-router-dom": "6.0.0",
                "framer-motion": "10.0.0"
            },
            {
                "react": "^18.0.0",
                "react-dom": "^18.0.0",
                "react-router-dom": "5.0.0" # Legacy conflict
            }
        ]

        # 3. State Tracking
        self.untouchable_packages = ["react", "react-dom"]
        self.max_steps = 10
        self.current_state = {}
        self.step_count = 0
        self.last_action = None # Fix #9: Prevent repeated actions

    def _generate_error_log(self, dependencies) -> str:
        """Checks current dependencies against the registry."""
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
        """Starts a new episode with a random broken package.json."""
        self.step_count = 0
        self.last_action = None
        
        # Select a random scenario to prevent overfitting
        self.current_state = random.choice(self.scenarios).copy()
        
        initial_errors = self._generate_error_log(self.current_state)
        
        return Observation(
            current_package_json=json.dumps(self.current_state, indent=2),
            npm_error_log=initial_errors,
            step_count=self.step_count
        )

    def step(self, action: Action):
        """Receives action, validates, updates state, and returns tuple of 4."""
        self.step_count += 1
        reward = 0
        done = False
        info = {} # Fix #1: Return 4 values

        # Fix #8: Check timeout first
        if self.step_count >= self.max_steps:
            reward -= 10
            done = True
            obs = Observation(json.dumps(self.current_state, indent=2), "TIMEOUT: Max steps reached.", self.step_count)
            return obs, reward, done, info

        pkg = action.package_to_update
        new_ver = action.new_version

        # Fix #9: Check for repeated useless actions
        if (pkg, new_ver) == self.last_action:
            reward -= 2
        self.last_action = (pkg, new_ver)

        # Fix #3: Validation for unknown packages
        if pkg not in self.current_state and pkg not in self.registry:
            reward -= 5
            obs = Observation(json.dumps(self.current_state, indent=2), f"Error: Cannot resolve package '{pkg}'.", self.step_count)
            return obs, reward, False, info

        # Fix #4: Validation for invalid version strings (must be DELETE or valid semver)
        if not isinstance(new_ver, str) or (not new_ver.startswith("^") and new_ver != "DELETE" and "." not in new_ver):
            reward -= 5
            obs = Observation(json.dumps(self.current_state, indent=2), f"Error: Invalid version format '{new_ver}'.", self.step_count)
            return obs, reward, False, info

        # Store old errors for progress check
        old_errors = self._generate_error_log(self.current_state)
        old_error_count = len(old_errors.split('\n')) if "SUCCESS" not in old_errors else 0

        # Apply Action
        if new_ver == "DELETE":
            if pkg in self.untouchable_packages:
                reward -= 100 # The Nuke Penalty
                done = True
                obs = Observation(json.dumps(self.current_state, indent=2), "FATAL: Cannot remove core framework.", self.step_count)
                return obs, reward, done, info
            else:
                self.current_state.pop(pkg, None)
        else:
            self.current_state[pkg] = new_ver

        # Fix #5: Check if they deleted everything to cheat
        if len(self.current_state) < 2:
            reward -= 100
            done = True
            obs = Observation(json.dumps(self.current_state, indent=2), "FATAL: Invalid dependency state.", self.step_count)
            return obs, reward, done, info

        # Step Penalty
        reward -= 1 

        # Evaluate new state
        error_log = self._generate_error_log(self.current_state)
        new_error_count = len(error_log.split('\n')) if "SUCCESS" not in error_log else 0
        
        # Fix #6: Partial Reward for progress
        if new_error_count < old_error_count and "SUCCESS" not in error_log:
            reward += 5

        # Check for ultimate success
        if "SUCCESS" in error_log:
            reward += 50
            done = True

        obs = Observation(
            current_package_json=json.dumps(self.current_state, indent=2),
            npm_error_log=error_log,
            step_count=self.step_count
        )
        
        return obs, reward, done, info # Returns the 4-tuple!


if __name__ == "__main__":
    print("--- STARTING RIGOROUS LOCAL ENVIRONMENT TEST ---")
    env = NPMResolverEnv()
    obs = env.reset()
    
    # Test 1: Invalid Format
    print("\n[TEST] Invalid Version Format (banana):")
    obs, reward, done, info = env.step(Action("framer-motion", "banana"))
    print(f"Reward: {reward} | Expected negative")
    
    # Test 2: Unknown Package
    print("\n[TEST] Unknown Package Injection:")
    obs, reward, done, info = env.step(Action("random-malware", "1.0.0"))
    print(f"Reward: {reward} | Expected negative")

    # Test 3: Cheating by Deletion
    print("\n[TEST] Deleting Core Framework:")
    obs, reward, done, info = env.step(Action("react", "DELETE"))
    print(f"Reward: {reward} | Expected -100")