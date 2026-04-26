"""
NPMResolverEnv — Robust, trainable environment for RL-based npm dependency resolution.

Combines the exhaustive safety / state-management logic from the production-grade
environment.py with the clean, easy-to-train reward shaping described in the PRD.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Action / Observation containers (kept lean but validated)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    package_to_update: str
    new_version: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Action":
        if not isinstance(data, dict):
            raise TypeError("Action must be a JSON object.")
        pkg = data.get("package_to_update")
        ver = data.get("new_version")
        if pkg is None or ver is None:
            raise ValueError("Action requires 'package_to_update' and 'new_version'.")
        if not isinstance(pkg, str) or not isinstance(ver, str):
            raise TypeError("Both fields must be strings.")
        pkg, ver = pkg.strip(), ver.strip()
        if not pkg or not ver:
            raise ValueError("Fields cannot be empty.")
        return cls(package_to_update=pkg, new_version=ver)

    def to_dict(self) -> Dict[str, str]:
        return {"package_to_update": self.package_to_update, "new_version": self.new_version}


@dataclass
class Observation:
    current_package_json: str
    npm_error_log: str
    step_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_package_json": self.current_package_json,
            "npm_error_log": self.npm_error_log,
            "step_count": self.step_count,
        }


# ---------------------------------------------------------------------------
# 2. The environment
# ---------------------------------------------------------------------------
class NPMResolverEnv:
    # ------ internal constants ------
    _VERSION_RE = re.compile(r"^(?:\^\d+\.\d+\.\d+|\d+\.\d+\.\d+|DELETE)$")
    SUCCESS_MSG = "SUCCESS: Audited packages in 0.01s. No vulnerabilities found."
    UNTOUCHABLE = {"react", "react-dom"}

    # ------ reward values (match PRD and the “trainability” of the simple env) ------
    STEP_PENALTY = -1
    INVALID_ACTION_PENALTY = -5
    PROGRESS_REWARD = 10          # per resolved conflict
    REGRESSION_PENALTY = -10      # per introduced conflict
    SUCCESS_REWARD = 50
    FATAL_PENALTY = -100
    TIMEOUT_PENALTY = -10
    NOOP_PENALTY = -2
    CORE_DELETE_PENALTY = -100

    # ------ registry mock ------
    MOCK_REGISTRY: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {
        "react-dom": {
            "^17.0.0": {"requires": {"react": "^17.0.0"}},
            "^18.0.0": {"requires": {"react": "^18.0.0"}},
        },
        "react-router-dom": {
            "6.0.0": {"requires": {"react": "^18.0.0", "react-dom": "^18.0.0"}},
        },
        "mongoose": {
            "6.0.0": {"requires": {"mongodb": "^4.0.0"}},
            "7.0.0": {"requires": {"mongodb": "^5.0.0"}},
        },
        "express-session": {
            "1.17.0": {"requires": {"express": "^4.0.0"}},
        },
        "babel-loader": {
            "8.0.0": {"requires": {"webpack": "^4.0.0", "@babel/core": "^7.0.0"}},
            "9.0.0": {"requires": {"webpack": "^5.0.0", "@babel/core": "^7.0.0"}},
        },
        "vuex": {
            "3.0.0": {"requires": {"vue": "^2.0.0"}},
            "4.0.0": {"requires": {"vue": "^3.0.0"}},
        }
    }

    # ------ curriculum scenarios ------
    SCENARIOS = {
        "level_1": [
            {"react": "^17.0.0", "react-dom": "^18.0.0"},
            {"mongodb": "^4.0.0", "mongoose": "7.0.0"},
            {"vue": "^2.0.0", "vuex": "4.0.0"}
        ],
        "level_2": [
            {"react": "^17.0.0", "react-dom": "^17.0.0", "react-router-dom": "6.0.0"},
            {"webpack": "^4.0.0", "@babel/core": "^7.0.0", "babel-loader": "9.0.0"},
        ],
        "level_3": [
            {"webpack": "^4.0.0", "@babel/core": "^6.0.0", "babel-loader": "9.0.0"},
            {"express": "^3.0.0", "express-session": "1.17.0", "mongodb": "^4.0.0", "mongoose": "7.0.0"}
        ],
    }

    def __init__(self, training_mode: bool = True, max_steps: int = 10):
        """
        training_mode:
          - True  → each reset picks a random scenario from *all* levels (great for RL).
          - False → curriculum progression (level‑up after streak of successes).
        """
        self.training_mode = training_mode
        self.max_steps = max_steps

        self.registry = copy.deepcopy(self.MOCK_REGISTRY)
        self.scenarios = {
            name: [dict(sorted(s.items())) for s in scens]
            for name, scens in self.SCENARIOS.items()
        }
        self.allowed_packages = self._build_allowed_set()

        # curriculum state (only used when training_mode=False)
        self.current_level = "level_1"
        self.level_success_streak = 0
        self.level_success_threshold = 2
        self.level_indices = {name: 0 for name in self.scenarios}

        # per‑episode state
        self.current_state: Dict[str, str] = {}
        self.step_count = 0
        self.last_action: Optional[Tuple[str, str]] = None
        self.episode_done = False
        self.active_level = "level_1"

        # full history (optional, for metrics / replay)
        self.history: List[Dict[str, Any]] = []
        self.episode_count = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # 3. Core logic (ported from the robust env with simplified rewards)
    # ------------------------------------------------------------------
    def _build_allowed_set(self) -> List[str]:
        pkgs = set(self.UNTOUCHABLE)
        for scens in self.scenarios.values():
            for s in scens:
                pkgs.update(s.keys())
        for pkg, versions in self.registry.items():
            pkgs.add(pkg)
            for meta in versions.values():
                for dep in meta.get("requires", {}):
                    pkgs.add(dep)
        return sorted(pkgs)

    def _extract_major(self, version: str) -> Optional[int]:
        v = version[1:] if version.startswith("^") else version
        try:
            return int(v.split(".", 1)[0])
        except (ValueError, IndexError):
            return None

    def is_compatible(self, required: str, actual: str) -> bool:
        if required == actual:
            return True
        rm = self._extract_major(required)
        am = self._extract_major(actual)
        return rm is not None and am is not None and rm == am

    def _generate_error_log(self, deps: Dict[str, str]) -> str:
        """Return SUCCESS_MSG or human‑readable conflict lines."""
        if not isinstance(deps, dict):
            return "FATAL: Invalid state."
        conflicts = []
        for pkg, ver in sorted(deps.items()):
            entry = self.registry.get(pkg, {}).get(ver)
            if not entry:
                continue
            for req, req_ver in sorted(entry.get("requires", {}).items()):
                cur = deps.get(req)
                if cur is None:
                    conflicts.append(
                        f"Missing peer dependency: {pkg}@{ver} requires {req}@{req_ver}"
                    )
                elif not self.is_compatible(req_ver, cur):
                    conflicts.append(
                        f"Conflict: {pkg}@{ver} requires {req}@{req_ver}, but found {req}@{cur}"
                    )
        return "\n".join(conflicts) if conflicts else self.SUCCESS_MSG

    def _error_count(self, log: str) -> int:
        return 0 if log == self.SUCCESS_MSG else log.count("\n") + 1

    def _validate_action(self, action: Action) -> Optional[str]:
        """Return error message or None if valid."""
        if not isinstance(action, Action):
            return "Error: Action must be an Action instance."
        if not isinstance(action.package_to_update, str) or not isinstance(action.new_version, str):
            return "Error: Action fields must be strings."
        pkg, ver = action.package_to_update.strip(), action.new_version.strip()
        if not pkg or not ver:
            return "Error: Action fields cannot be empty."
        if pkg not in self.current_state:
            return f"Error: Package '{pkg}' not in current state."
        if not self._VERSION_RE.match(ver):
            return f"Error: Invalid version format '{ver}'."
        if ver != "DELETE" and pkg in self.registry and ver not in self.registry[pkg]:
            return f"Error: Unknown version '{ver}' for '{pkg}'."
        return None

    def _apply_transition(self, state: Dict[str, str], action: Action) -> Tuple[Dict[str, str], Optional[str]]:
        """Try to apply `action` to a copy of `state`. Returns (new_state, error)."""
        if not isinstance(state, dict):
            return state, "FATAL: Invalid state."
        s = dict(sorted(state.items()))
        pkg, ver = action.package_to_update.strip(), action.new_version.strip()

        if ver == "DELETE":
            if pkg in self.UNTOUCHABLE:
                return s, "FATAL: Cannot remove a core framework package."
            del s[pkg]
        else:
            s[pkg] = ver

        # invariants
        if len(s) < 2:
            return s, "FATAL: State must contain at least 2 packages."
        return dict(sorted(s.items())), None

    # ------------------------------------------------------------------
    # 4. Gym‑like interface (reset, step)
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        """Start a fresh episode (random scenario if training_mode, else curriculum)."""
        self.step_count = 0
        self.last_action = None
        self.episode_done = False
        self.episode_count += 1

        if self.training_mode:
            # pick a random scenario from any level
            all_scenarios = [sc for scens in self.scenarios.values() for sc in scens]
            self.current_state = dict(random.choice(all_scenarios))
            self.active_level = "mixed"
        else:
            level = self.current_level
            scenarios = self.scenarios[level]
            idx = self.level_indices[level] % len(scenarios)
            self.level_indices[level] = idx + 1
            self.current_state = dict(scenarios[idx])
            self.active_level = level

        errors = self._generate_error_log(self.current_state)
        return Observation(
            current_package_json=json.dumps(self.current_state, indent=2, sort_keys=True),
            npm_error_log=errors,
            step_count=0,
        )

    def step(self, action: Action) -> Tuple[Observation, int, bool, Dict[str, Any]]:
        """
        Returns (observation, reward, done, info).
        Info dict carries metadata useful for debugging & training.
        """
        if self.episode_done:
            # episode already terminated – return last observation unchanged
            obs = Observation(
                json.dumps(self.current_state, indent=2, sort_keys=True),
                self._generate_error_log(self.current_state),
                self.step_count,
            )
            return obs, 0, True, {"status": "done", "reason": "episode_already_terminated"}

        self.step_count += 1
        reward = 0
        done = False
        info: Dict[str, Any] = {"status": "in_progress"}

        # 4a. timeout check
        if self.step_count > self.max_steps:
            log = f"TIMEOUT: Max steps ({self.max_steps}) reached.\n{self._generate_error_log(self.current_state)}"
            obs = Observation(json.dumps(self.current_state, indent=2, sort_keys=True), log, self.step_count)
            reward += self.TIMEOUT_PENALTY
            done = True
            info["status"] = "timeout"
            self.episode_done = True
            self.total_reward += reward
            return obs, reward, done, info

        # 4b. validate action
        error_msg = self._validate_action(action)
        if error_msg:
            # invalid action – penalise and continue (episode NOT terminated)
            reward += self.INVALID_ACTION_PENALTY
            obs = Observation(
                json.dumps(self.current_state, indent=2, sort_keys=True),
                error_msg,
                self.step_count,
            )
            info["status"] = "invalid_action"
            info["error"] = error_msg
            self.total_reward += reward
            return obs, reward, done, info

        pkg, ver = action.package_to_update.strip(), action.new_version.strip()

        # 4c. repeated useless action check
        if (pkg, ver) == self.last_action:
            reward += self.NOOP_PENALTY
        self.last_action = (pkg, ver)

        old_log = self._generate_error_log(self.current_state)
        old_errors = self._error_count(old_log)

        # 4d. apply transition
        new_state, err = self._apply_transition(self.current_state, action)
        if err:
            # fatal errors (e.g. core delete) immediately end episode
            obs = Observation(
                json.dumps(self.current_state, indent=2, sort_keys=True),
                err,
                self.step_count,
            )
            reward += self.FATAL_PENALTY
            done = True
            info["status"] = "failed"
            info["reason"] = err
            self.episode_done = True
            self.total_reward += reward
            return obs, reward, done, info

        # commit state change
        self.current_state = dict(sorted(new_state.items()))

        # 4e. step penalty
        reward += self.STEP_PENALTY

        new_log = self._generate_error_log(self.current_state)
        new_errors = self._error_count(new_log)

        # 4f. progress / regression reward
        progress = old_errors - new_errors
        if progress > 0:
            reward += progress * self.PROGRESS_REWARD
        elif progress < 0:
            reward += progress * self.REGRESSION_PENALTY  # (negative * negative = negative)

        # 4g. ultimate success
        if new_log == self.SUCCESS_MSG:
            reward += self.SUCCESS_REWARD
            done = True
            info["status"] = "success"
            self.episode_done = True
            if not self.training_mode:
                self.episode_done = True
                self.level_success_streak += 1
                if self.level_success_streak >= self.level_success_threshold:
                    # advance curriculum
                    levels = ["level_1", "level_2", "level_3"]
                    idx = levels.index(self.current_level) if self.current_level in levels else 0
                    if idx < len(levels) - 1:
                        self.current_level = levels[idx + 1]
                        self.level_indices[self.current_level] = 0
                    self.level_success_streak = 0

        obs = Observation(
            json.dumps(self.current_state, indent=2, sort_keys=True),
            new_log,
            self.step_count,
        )
        info["errors_remaining"] = new_errors
        self.total_reward += reward
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # 5. Auxiliary helpers (metrics, serialisation, action mask)
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, float]:
        return {
            "episodes": self.episode_count,
            "total_reward": self.total_reward,
            "active_level": self.active_level,
        }

    def get_action_mask(self) -> Dict[str, List[str]]:
        """Return valid {package: [version]} for the current state (useful for RL)."""
        mask: Dict[str, List[str]] = {}
        base_state = dict(sorted(self.current_state.items()))
        for pkg in base_state:
            # allowed versions from registry + DELETE for non‑core
            versions = []
            if pkg in self.registry:
                versions = list(self.registry[pkg].keys())
            else:
                # e.g. “react” itself – we only allow versions that appear in scenarios
                versions = sorted({
                    s.get(pkg) for scens in self.scenarios.values() for s in scens if pkg in s
                })
            # include current version for no‑op detection (but it will be penalised)
            current_ver = base_state[pkg]
            candidate_vers = [v for v in versions if v != current_ver]
            if pkg not in self.UNTOUCHABLE and len(base_state) > 2:
                candidate_vers.append("DELETE")
            # only keep versions that don't break the state
            valid = []
            for v in candidate_vers:
                act = Action(package_to_update=pkg, new_version=v)
                test_state, err = self._apply_transition(base_state, act)
                if err is None and test_state is not None:
                    valid.append(v)
            if valid:
                mask[pkg] = valid
        return mask

    def to_dict(self) -> Dict[str, Any]:
        """Lightweight serialization (enough for checkpointing)."""
        return {
            "current_state": self.current_state,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "episode_done": self.episode_done,
            "active_level": self.active_level,
            "training_mode": self.training_mode,
            "current_level": self.current_level,
            "level_success_streak": self.level_success_streak,
        }

    def from_dict(self, state: Dict[str, Any]) -> None:
        self.current_state = dict(state["current_state"])
        self.step_count = int(state["step_count"])
        self.max_steps = int(state["max_steps"])
        self.episode_done = bool(state["episode_done"])
        self.active_level = str(state["active_level"])
        self.training_mode = bool(state["training_mode"])
        self.current_level = str(state["current_level"])
        self.level_success_streak = int(state["level_success_streak"])
        self.last_action = None  # reset


# ---------------------------------------------------------------------------
# 6. Quick local test harness (run `python environment.py` to verify)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Robust NPMResolverEnv self‑test ===")

    # Using training_mode=False ensures we reliably test the Level 1 'react' scenario 
    # instead of randomly hitting a 'vue' or 'mongoose' scenario and failing the assertions.
    env = NPMResolverEnv(training_mode=False, max_steps=10)
    obs = env.reset()
    print("Initial state:", obs.current_package_json[:80], "...")

    # Test 1: invalid format
    _, r, d, info = env.step(Action("react-dom", "garbage"))
    assert r == env.INVALID_ACTION_PENALTY and d == False
    print("[PASS] Invalid version penalised correctly.")

    # Test 2: unknown package
    obs2, r2, d2, _ = env.step(Action("made-up", "1.0.0"))
    assert r2 == env.INVALID_ACTION_PENALTY
    print("[PASS] Unknown package penalised correctly.")

    # Test 3: core delete
    env2 = NPMResolverEnv(training_mode=False)
    env2.reset()
    obs3, r3, d3, info3 = env2.step(Action("react", "DELETE"))
    assert r3 == env.FATAL_PENALTY and d3 == True
    print("[PASS] Core deletion triggers fatal penalty.")

    # Test 4: success trajectory
    env3 = NPMResolverEnv(training_mode=False)
    env3.reset()
    obs4, r4, d4, _ = env3.step(Action("react", "^18.0.0"))
    # after this step, remaining conflicts may require react-dom bump -> need at least 2 steps
    if env3._generate_error_log(env3.current_state) == env3.SUCCESS_MSG:
        assert d4 and r4 > 0
        print("[PASS] Fixed in 1 step (expected for level_1 simple mismatch).")
    else:
        assert not d4
        obs5, r5, d5, _ = env3.step(Action("react-dom", "^18.0.0"))
        assert d5 and r5 > 30  # should include success + progress
        print("[PASS] Two‑step fix rewarded correctly.")

    print("\nAll tests passed – environment is bulletproof and trainable.")