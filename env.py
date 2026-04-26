"""
NPMResolverEnv — Universal, Robust, Procedurally Generated Environment.

Combines exhaustive state-management and reward shaping with 
an infinite scenario generator to prevent LLM overfitting.
"""
from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Action / Observation containers (Robust)
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
# 2. The Universal Environment
# ---------------------------------------------------------------------------
class NPMResolverEnv:
    # ------ internal constants ------
    _VERSION_RE = re.compile(r"^(?:\^\d+\.\d+\.\d+|\d+\.\d+\.\d+|DELETE)$")
    SUCCESS_MSG = "SUCCESS: Audited packages in 0.01s. No vulnerabilities found."

    # ------ reward values ------
    STEP_PENALTY = -1
    INVALID_ACTION_PENALTY = -5
    PROGRESS_REWARD = 10          
    REGRESSION_PENALTY = -10      
    SUCCESS_REWARD = 50
    FATAL_PENALTY = -100
    TIMEOUT_PENALTY = -10
    NOOP_PENALTY = -2

    # ------ Infinite Generator Vocabulary ------
    VOCAB = [
        "react", "express", "lodash", "mongoose", "axios", "webpack", 
        "jest", "chalk", "next", "tailwindcss", "typescript", "eslint",
        "alpha-lib", "beta-core", "gamma-utils", "delta-auth", "omega-router"
    ]

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.current_state: Dict[str, str] = {}
        self.active_rule: Dict[str, str] = {}
        self.step_count = 0
        self.last_action: Optional[Tuple[str, str]] = None
        self.episode_done = False
        self.untouchable_pkg = "" # The generated root package

    # ------------------------------------------------------------------
    # 3. Core logic 
    # ------------------------------------------------------------------
    def _generate_error_log(self, deps: Dict[str, str]) -> str:
        """Evaluates the current state against the procedural rule."""
        if not self.active_rule:
            return "FATAL: Invalid rule state."
            
        root = self.active_rule["root"]
        peer = self.active_rule["peer"]
        req_ver = self.active_rule["req_peer_ver"]
        
        # Prevent the AI from cheating by deleting the root package
        if root not in deps:
            return f"FATAL: Core package {root} was removed."

        if peer not in deps:
            return f"Missing peer dependency: {root}@{self.active_rule['root_ver']} requires {peer}@{req_ver}"
        elif deps[peer] != req_ver:
            return f"Conflict: {root}@{self.active_rule['root_ver']} requires {peer}@{req_ver}, but found {deps[peer]}"
            
        return self.SUCCESS_MSG

    def _error_count(self, log: str) -> int:
        return 0 if log == self.SUCCESS_MSG else log.count("\n") + 1

    def _validate_action(self, action: Action) -> Optional[str]:
        if not isinstance(action, Action):
            return "Error: Action must be an Action instance."
        if not isinstance(action.package_to_update, str) or not isinstance(action.new_version, str):
            return "Error: Action fields must be strings."
        pkg, ver = action.package_to_update.strip(), action.new_version.strip()
        
        if not pkg or not ver:
            return "Error: Action fields cannot be empty."
        if not self._VERSION_RE.match(ver):
            return f"Error: Invalid version format '{ver}'."
        return None

    def _apply_transition(self, state: Dict[str, str], action: Action) -> Tuple[Dict[str, str], Optional[str]]:
        if not isinstance(state, dict):
            return state, "FATAL: Invalid state."
        s = dict(sorted(state.items()))
        pkg, ver = action.package_to_update.strip(), action.new_version.strip()

        if ver == "DELETE":
            if pkg == self.untouchable_pkg:
                return s, f"FATAL: Cannot remove core package '{pkg}'."
            del s[pkg]
        else:
            s[pkg] = ver

        if len(s) < 1:
            return s, "FATAL: State must contain at least 1 package."
        return dict(sorted(s.items())), None

    # ------------------------------------------------------------------
    # 4. Interface (reset, step)
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        """Starts a fresh procedurally generated episode."""
        self.step_count = 0
        self.last_action = None
        self.episode_done = False
        
        # 1. Procedural Generation
        pkgs = random.sample(self.VOCAB, 3)
        root_pkg, peer_pkg, extra_pkg = pkgs
        
        bad_version = f"^{random.randint(1, 5)}.{random.randint(0, 5)}.0"
        req_version = f"^{random.randint(6, 12)}.{random.randint(0, 5)}.0"
        root_version = f"{random.randint(1, 4)}.0.0"
        
        self.current_state = {
            root_pkg: root_version,
            peer_pkg: bad_version,
            extra_pkg: "^1.0.0"
        }
        
        self.active_rule = {
            "root": root_pkg,
            "root_ver": root_version,
            "peer": peer_pkg,
            "req_peer_ver": req_version
        }
        self.untouchable_pkg = root_pkg
        
        # 2. Randomly simulate Missing vs Conflict
        if random.random() > 0.5:
            del self.current_state[peer_pkg]
            
        errors = self._generate_error_log(self.current_state)
        
        return Observation(
            current_package_json=json.dumps(self.current_state, indent=2, sort_keys=True),
            npm_error_log=errors,
            step_count=0,
        )

    def step(self, action: Action) -> Tuple[Observation, int, bool, Dict[str, Any]]:
        if self.episode_done:
            obs = Observation(
                json.dumps(self.current_state, indent=2, sort_keys=True),
                self._generate_error_log(self.current_state),
                self.step_count,
            )
            return obs, 0, True, {"status": "done"}

        self.step_count += 1
        reward = 0
        done = False
        info: Dict[str, Any] = {"status": "in_progress"}

        # 4a. Timeout
        if self.step_count > self.max_steps:
            log = f"TIMEOUT: Max steps ({self.max_steps}) reached."
            obs = Observation(json.dumps(self.current_state, indent=2, sort_keys=True), log, self.step_count)
            self.episode_done = True
            return obs, self.TIMEOUT_PENALTY, True, info

        # 4b. Validate Action
        error_msg = self._validate_action(action)
        if error_msg:
            reward += self.INVALID_ACTION_PENALTY
            obs = Observation(
                json.dumps(self.current_state, indent=2, sort_keys=True),
                error_msg,
                self.step_count,
            )
            return obs, reward, done, info

        pkg, ver = action.package_to_update.strip(), action.new_version.strip()

        # 4c. Repetition penalty
        if (pkg, ver) == self.last_action:
            reward += self.NOOP_PENALTY
        self.last_action = (pkg, ver)

        old_log = self._generate_error_log(self.current_state)
        old_errors = self._error_count(old_log)

        # 4d. Transition
        new_state, err = self._apply_transition(self.current_state, action)
        if err:
            obs = Observation(json.dumps(self.current_state, indent=2, sort_keys=True), err, self.step_count)
            self.episode_done = True
            return obs, self.FATAL_PENALTY, True, info

        self.current_state = dict(sorted(new_state.items()))

        # 4e. Step Penalty
        reward += self.STEP_PENALTY

        new_log = self._generate_error_log(self.current_state)
        new_errors = self._error_count(new_log)

        # 4f. Progress Reward
        progress = old_errors - new_errors
        if progress > 0:
            reward += progress * self.PROGRESS_REWARD
        elif progress < 0:
            reward += progress * self.REGRESSION_PENALTY 

        # 4g. Success
        if new_log == self.SUCCESS_MSG:
            reward += self.SUCCESS_REWARD
            done = True
            self.episode_done = True
            info["status"] = "success"

        obs = Observation(
            json.dumps(self.current_state, indent=2, sort_keys=True),
            new_log,
            self.step_count,
        )
        
        return obs, reward, done, info