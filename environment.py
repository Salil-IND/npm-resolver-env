"""Production-grade RL environment for simplified npm dependency resolution.

This module defines a deterministic Markov Decision Process for dependency
repair. The MDP is intentionally simplified so agents can focus on
decision-making rather than npm implementation details.

MDP definition:
- State: a JSON-serializable dependency map plus a deterministic audit log.
- Action: update one package version or delete one non-core package.
- Transition: apply exactly one validated dependency edit atomically.
- Reward: step cost plus shaped progress, regression, deletion, success, and
  timeout signals.
- Termination: success, timeout, fatal anti-cheat violation, or internal
  fail-safe recovery.
"""

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Action:
    """Represents a single dependency update decision made by an agent.

    Inputs:
        package_to_update: Name of the package to modify.
        new_version: Target version string or the literal "DELETE".

    Outputs:
        Immutable action object suitable for `NPMResolverEnv.step()`.

    Behavior:
        Provides JSON-friendly construction and serialization helpers.
    """

    package_to_update: str
    new_version: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Action":
        """Build an Action from a JSON-compatible dictionary.

        Inputs:
            data: Dictionary containing `package_to_update` and `new_version`.

        Outputs:
            A validated `Action` instance.

        Behavior:
            Raises `TypeError` or `ValueError` when fields are missing or invalid.
        """
        if not isinstance(data, dict):
            raise TypeError("Action JSON payload must be a dictionary.")

        package = data.get("package_to_update")
        version = data.get("new_version")

        if package is None or version is None:
            raise ValueError("Action JSON payload must include both required fields.")

        if not isinstance(package, str) or not isinstance(version, str):
            raise TypeError("Action fields must be strings.")

        package = package.strip()
        version = version.strip()

        if not package or not version:
            raise ValueError("Action fields cannot be empty.")

        return cls(package_to_update=package, new_version=version)

    def to_dict(self) -> Dict[str, str]:
        """Serialize an Action into a JSON-friendly dictionary.

        Inputs:
            None.

        Outputs:
            Dictionary representation of the action.

        Behavior:
            Returns a compact payload suitable for logging or transport.
        """
        return {
            "package_to_update": self.package_to_update,
            "new_version": self.new_version,
        }


@dataclass
class Observation:
    """Represents the observable environment state returned to an agent.

    Inputs:
        current_package_json: JSON snapshot of dependency state.
        npm_error_log: Human-readable dependency audit message.
        step_count: Current episode step number.

    Outputs:
        Structured observation payload.

    Behavior:
        Supports JSON-friendly serialization through `to_dict()`.
    """

    current_package_json: str
    npm_error_log: str
    step_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize an Observation into a JSON-friendly dictionary.

        Inputs:
            None.

        Outputs:
            Dictionary representation of the observation.

        Behavior:
            Preserves the exact strings returned by the environment.
        """
        return {
            "current_package_json": self.current_package_json,
            "npm_error_log": self.npm_error_log,
            "step_count": self.step_count,
        }


class NPMResolverEnv:
    """OpenEnv-compatible environment for RL-driven npm dependency resolution.

    Inputs:
        None during construction.

    Outputs:
        Stateful environment with `reset()` and `step()` methods.

    Behavior:
        Simulates dependency conflicts, curriculum difficulty, reward shaping,
        anti-cheat protection, and robust serialization for RL integrations.
    """

    VERSION_PATTERN = re.compile(r"^(?:\^\d+\.\d+\.\d+|\d+\.\d+\.\d+|DELETE)$")

    def __init__(self) -> None:
        """Initialize environment configuration, curriculum, and runtime state.

        Inputs:
            None.

        Outputs:
            Ready-to-use `NPMResolverEnv` instance.

        Behavior:
            Sets up registry metadata, curriculum levels, anti-cheat rules,
            episode tracking, and debug logging buffers.
        """
        self.current_state: Dict[str, str] = {}
        self.step_count = 0
        self.max_steps = 10
        self.last_action: Optional[Action] = None
        self.last_error_count = 0
        self.last_error_log = ""
        self.episode_done = False
        self.episode_status = "in_progress"
        self.current_scenario_index = 0
        self.current_episode_reward = 0
        self._episode_metrics_committed = False
        self.debug = True
        self.debug_logs: List[str] = []
        self.demo_mode = True
        self.demo_trace: List[Dict[str, Any]] = []
        self.current_level = "level_1"
        self.active_level = "level_1"
        self.level_order = ["level_1", "level_2", "level_3"]
        self.success_count = 0
        self.level_success_streak = 0
        self.level_success_threshold = 2
        self.total_steps = 0
        self.episode_count = 0
        self.total_reward = 0
        self.history: List[Dict[str, Any]] = []
        self.reward_history: List[int] = []
        self.episode_reward_history: List[int] = []
        self.episode_step_history: List[int] = []
        self.episode_status_history: List[str] = []
        self.untouchable_packages = ["react", "react-dom"]
        self.success_message = "SUCCESS: Audited packages in 0.01s. No vulnerabilities found."
        self.timeout_message = "TIMEOUT: Maximum steps reached without resolution."
        self.step_penalty = -1
        self.progress_reward_per_error = 10
        self.regression_penalty_per_error = 10
        self.stalled_progress_penalty = 0
        self.no_op_penalty = -2
        self.invalid_action_penalty = -5
        self.deletion_penalty = -6
        self.timeout_penalty = -10
        self.timeout_reward_cap = -1
        self.success_reward = 50
        self.fatal_penalty = -100
        self.registry = {
            "react-dom": {
                "^17.0.0": {"requires": {"react": "^17.0.0"}},
                "^18.0.0": {"requires": {"react": "^18.0.0"}},
            },
            "react-router-dom": {
                "5.0.0": {
                    "requires": {
                        "react": "^16.8.0",
                        "react-dom": "^16.8.0",
                    }
                },
                "6.0.0": {
                    "requires": {
                        "react": "^18.0.0",
                        "react-dom": "^18.0.0",
                    }
                },
            },
            "framer-motion": {
                "4.0.0": {"requires": {"react": "^17.0.0"}},
                "10.0.0": {"requires": {"react": "^18.0.0"}},
            },
        }
        self.level_descriptions = {
            "level_1": "Single dependency mismatch with one obvious fix.",
            "level_2": "Cascading dependency conflicts requiring coordinated updates.",
            "level_3": "Multi-package resolution with competing dependency expectations.",
        }
        self.levels = {
            "level_1": [
                {
                    "react": "^17.0.0",
                    "react-dom": "^18.0.0",
                },
                {
                    "react": "^18.0.0",
                    "react-dom": "^17.0.0",
                }
            ],
            "level_2": [
                {
                    "react": "^17.0.0",
                    "react-dom": "^17.0.0",
                    "react-router-dom": "6.0.0",
                },
                {
                    "react": "^17.0.0",
                    "react-dom": "^18.0.0",
                    "react-router-dom": "6.0.0",
                },
                {
                    "react": "^17.0.0",
                    "react-dom": "^17.0.0",
                    "framer-motion": "10.0.0",
                }
            ],
            "level_3": [
                {
                    "react": "^18.0.0",
                    "react-dom": "^18.0.0",
                    "react-router-dom": "5.0.0",
                    "framer-motion": "4.0.0",
                },
                {
                    "react": "^17.0.0",
                    "react-dom": "^17.0.0",
                    "react-router-dom": "6.0.0",
                    "framer-motion": "10.0.0",
                },
                {
                    "react": "^18.0.0",
                    "react-dom": "^17.0.0",
                    "react-router-dom": "5.0.0",
                    "framer-motion": "10.0.0",
                }
            ],
        }
        self.level_indices = {level_name: 0 for level_name in self.levels}
        self.demo_scenario = {
            "react": "^17.0.0",
            "react-dom": "^17.0.0",
            "react-router-dom": "6.0.0",
        }
        self._refresh_cached_metadata()

    def _log_debug(self, message: str) -> None:
        """Record a bounded internal debug message when debug mode is enabled.

        Inputs:
            message: Debug string to append to the internal log buffer.

        Outputs:
            None.

        Behavior:
            Keeps the last 100 messages without printing to stdout.
        """
        if not self.debug:
            return

        self.debug_logs.append(message)
        if len(self.debug_logs) > 100:
            self.debug_logs.pop(0)

    def _format_error(self, prefix: str, message: str) -> str:
        """Create a standardized human-readable error string.

        Inputs:
            prefix: Error category such as `Error`, `FATAL`, or `TIMEOUT`.
            message: Human-readable explanation.

        Outputs:
            Standardized formatted error message.

        Behavior:
            Ensures consistent error strings across all environment paths.
        """
        return f"{prefix}: {message}"

    def _count_errors(self, error_log: str) -> int:
        """Count logical conflict lines in an error log.

        Inputs:
            error_log: Audit or failure string generated by the environment.

        Outputs:
            Integer number of active errors.

        Behavior:
            Treats the success message as zero errors and counts non-empty
            multi-line logs in O(n) over the string length.
        """
        if not error_log or error_log == self.success_message:
            return 0

        if error_log.startswith(self.timeout_message):
            _, _, remainder = error_log.partition("\n")
            if not remainder:
                return 0
            return self._count_errors(remainder)

        return error_log.count("\n") + 1

    def _build_allowed_packages(self) -> List[str]:
        """Build the deterministic list of package names allowed in the environment.

        Inputs:
            None.

        Outputs:
            Sorted list of allowed package names.

        Behavior:
            Combines packages from curriculum scenarios, demo scenarios, registry
            metadata, and peer requirements to prevent fake-package injection.
        """
        allowed_packages = set(self.untouchable_packages)

        for scenario_group in self.levels.values():
            for scenario in scenario_group:
                allowed_packages.update(scenario.keys())

        allowed_packages.update(self.demo_scenario.keys())
        allowed_packages.update(self.registry.keys())

        for package_versions in self.registry.values():
            for metadata in package_versions.values():
                allowed_packages.update(metadata.get("requires", {}).keys())

        return sorted(allowed_packages)

    def _normalize_registry(
        self, registry_payload: object
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
        """Normalize a registry payload into the internal registry structure.

        Inputs:
            registry_payload: Candidate registry payload.

        Outputs:
            Normalized registry dictionary.

        Behavior:
            Validates package names, version maps, and dependency requirements so
            external state restoration stays safe and deterministic.
        """
        if not isinstance(registry_payload, dict) or not registry_payload:
            raise ValueError("Registry payload must be a non-empty dictionary.")

        normalized_registry: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
        for package_name, versions in registry_payload.items():
            if not isinstance(package_name, str) or package_name.strip() == "":
                raise ValueError("Registry package names must be non-empty strings.")
            if not isinstance(versions, dict) or not versions:
                raise ValueError("Each registry package must define at least one version.")

            normalized_versions: Dict[str, Dict[str, Dict[str, str]]] = {}
            for version_name, metadata in versions.items():
                if not isinstance(version_name, str) or version_name.strip() == "":
                    raise ValueError("Registry version names must be non-empty strings.")
                if not isinstance(metadata, dict):
                    raise ValueError("Registry version metadata must be dictionaries.")

                requires = metadata.get("requires", {})
                if not isinstance(requires, dict):
                    raise ValueError("Registry requires entries must be dictionaries.")

                normalized_requires: Dict[str, str] = {}
                for dependency_name, dependency_version in requires.items():
                    if not isinstance(dependency_name, str) or dependency_name.strip() == "":
                        raise ValueError("Dependency names must be non-empty strings.")
                    if not isinstance(dependency_version, str) or dependency_version.strip() == "":
                        raise ValueError("Dependency versions must be non-empty strings.")
                    normalized_requires[dependency_name] = dependency_version

                normalized_versions[version_name] = {"requires": normalized_requires}

            normalized_registry[package_name] = normalized_versions

        return normalized_registry

    def _normalize_levels(self, levels_payload: object) -> Dict[str, List[Dict[str, str]]]:
        """Normalize curriculum levels into the internal deterministic structure.

        Inputs:
            levels_payload: Candidate curriculum payload.

        Outputs:
            Normalized mapping of level names to scenario lists.

        Behavior:
            Validates level names and scenario payloads while preserving scenario
            order for deterministic replay.
        """
        if not isinstance(levels_payload, dict) or not levels_payload:
            raise ValueError("Levels payload must be a non-empty dictionary.")

        normalized_levels: Dict[str, List[Dict[str, str]]] = {}
        for level_name, scenarios in levels_payload.items():
            if not isinstance(level_name, str) or level_name.strip() == "":
                raise ValueError("Level names must be non-empty strings.")
            if not isinstance(scenarios, list) or not scenarios:
                raise ValueError("Each level must define at least one scenario.")

            normalized_scenarios: List[Dict[str, str]] = []
            for scenario in scenarios:
                if not self._is_well_formed_state(
                    scenario, require_allowed_packages=False
                ):
                    raise ValueError("Each scenario must be a valid non-empty dictionary.")
                normalized_scenario = {
                    str(package_name): str(package_version)
                    for package_name, package_version in scenario.items()
                }
                normalized_scenarios.append(normalized_scenario)

            normalized_levels[level_name] = normalized_scenarios

        return normalized_levels

    def _refresh_cached_metadata(self) -> None:
        """Refresh cached metadata after registry or curriculum changes.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Recomputes allowed package names and ensures deterministic scenario
            indices exist for every curriculum level.
        """
        self.allowed_packages = self._build_allowed_packages()
        current_indices = getattr(self, "level_indices", {})
        self.level_indices = {
            level_name: max(0, int(current_indices.get(level_name, 0)))
            for level_name in self.levels
        }

    def _is_allowed_package_name(self, package_name: str) -> bool:
        """Check whether a package name is part of the supported environment.

        Inputs:
            package_name: Candidate package name.

        Outputs:
            Boolean validity flag.

        Behavior:
            Prevents fake packages from being inserted into environment state.
        """
        allowed_packages = getattr(self, "allowed_packages", [])
        return package_name in allowed_packages

    def _is_well_formed_state(
        self, state: object, require_allowed_packages: bool = True
    ) -> bool:
        """Validate the structural integrity of a dependency mapping.

        Inputs:
            state: Candidate state object.
            require_allowed_packages: Whether package names must already be known
                to the environment.

        Outputs:
            Boolean validity flag.

        Behavior:
            Validates dictionary shape, string keys and values, and optionally
            restricts package names to the environment package universe.
        """
        if not isinstance(state, dict) or not state:
            return False

        for package_name, package_version in state.items():
            if not isinstance(package_name, str) or package_name.strip() == "":
                return False
            if not isinstance(package_version, str) or package_version.strip() == "":
                return False
            if require_allowed_packages and not self._is_allowed_package_name(
                package_name
            ):
                return False

        return True

    def _is_valid_state(self, state: object) -> bool:
        """Validate that a dependency state is a non-empty string dictionary.

        Inputs:
            state: Candidate state object.

        Outputs:
            Boolean validity flag.

        Behavior:
            Rejects malformed states before they can corrupt transitions.
        """
        return self._is_well_formed_state(state, require_allowed_packages=True)

    def _safe_json_dumps(self, payload: object) -> str:
        """Serialize a payload into stable JSON without raising outward errors.

        Inputs:
            payload: Serializable Python object.

        Outputs:
            JSON string.

        Behavior:
            Falls back to `{}` when serialization fails.
        """
        try:
            return json.dumps(payload, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return json.dumps({}, indent=2, sort_keys=True)

    def _get_observable_state(self) -> Dict[str, str]:
        """Return the safe public view of the current dependency state.

        Inputs:
            None.

        Outputs:
            Copy of the current state or an empty dictionary.

        Behavior:
            Prevents malformed internal state from leaking into observations.
        """
        if self._is_valid_state(self.current_state):
            return dict(self.current_state)

        return {}

    def _build_observation(self, error_log: str) -> Observation:
        """Construct a stable observation from the current state and log.

        Inputs:
            error_log: Error or success message for the current timestep.

        Outputs:
            `Observation` instance.

        Behavior:
            Guarantees valid JSON in `current_package_json`.
        """
        observable_state = self._get_observable_state()
        return Observation(
            current_package_json=self._safe_json_dumps(observable_state),
            npm_error_log=error_log,
            step_count=self.step_count,
        )

    def _build_info(
        self,
        status: str,
        errors_remaining: int,
        old_error_count: Optional[int] = None,
        new_error_count: Optional[int] = None,
    ) -> dict:
        """Build a framework-friendly info dictionary for diagnostics.

        Inputs:
            status: Episode transition status string.
            errors_remaining: Current number of outstanding dependency issues.
            old_error_count: Optional pre-action error count.
            new_error_count: Optional post-action error count.

        Outputs:
            Dictionary containing stable debugging metadata.

        Behavior:
            Includes curriculum metadata and optional debug counters.
        """
        info = {
            "status": status,
            "step_count": self.step_count,
            "errors_remaining": errors_remaining,
            "current_level": self.active_level,
            "level_description": self.level_descriptions.get(
                self.active_level, "Custom scenario"
            ),
            "scenario_index": self.current_scenario_index,
            "success_count": self.success_count,
            "current_episode_reward": self.current_episode_reward,
        }

        if old_error_count is not None:
            info["old_error_count"] = old_error_count

        if new_error_count is not None:
            info["new_error_count"] = new_error_count

        if self.debug:
            info["debug_log_count"] = len(self.debug_logs)

        return info

    def _record_step_metrics(
        self,
        action: Optional[Action],
        reward: int,
        done: bool,
        status: str,
        error_log: str,
        old_error_count: Optional[int] = None,
        new_error_count: Optional[int] = None,
    ) -> None:
        """Record lightweight per-step metrics without altering RL logic.

        Inputs:
            action: Action taken for the current transition, if any.
            reward: Reward emitted by the transition.
            done: Terminal flag for the transition.
            status: Transition status string.
            error_log: Final log generated for the transition.

        Outputs:
            None.

        Behavior:
            Updates cumulative counters and appends a JSON-friendly step record.
        """
        self.total_steps += 1
        self.total_reward += reward
        self.current_episode_reward += reward
        self.reward_history.append(reward)
        self.history.append(
            {
                "episode": self.episode_count,
                "step": self.step_count,
                "level": self.active_level,
                "scenario_index": self.current_scenario_index,
                "state": dict(self._get_observable_state()),
                "action": None if action is None else action.to_dict(),
                "reward": reward,
                "done": done,
                "status": status,
                "error_log": error_log,
                "old_error_count": old_error_count,
                "new_error_count": new_error_count,
            }
        )

        if self.demo_mode:
            action_payload = None if action is None else action.to_dict()
            transition_label = "No action recorded."
            if action is not None:
                transition_label = (
                    f"Updated {action.package_to_update} to {action.new_version}"
                )
            summary = (
                f"Step {self.step_count}: {transition_label}. "
                f"Reward {reward:+d}. Status: {status.upper()}."
            )
            if old_error_count is not None and new_error_count is not None:
                summary = (
                    f"{summary} Errors: {old_error_count} -> {new_error_count}."
                )
            self.demo_trace.append(
                {
                    "step": self.step_count,
                    "level": self.active_level,
                    "level_description": self.level_descriptions.get(
                        self.active_level, "Custom scenario"
                    ),
                    "scenario_index": self.current_scenario_index,
                    "state": dict(self._get_observable_state()),
                    "state_json": self._safe_json_dumps(self._get_observable_state()),
                    "action": action_payload,
                    "reward": reward,
                    "error": error_log,
                    "status": status,
                    "remaining_errors": self._count_errors(error_log),
                    "errors_before": old_error_count,
                    "errors_after": new_error_count,
                    "transition": transition_label,
                    "summary": summary,
                    "resolved": status == "success",
                    "final": done,
                }
            )

    def _record_completed_episode(self, status: str) -> None:
        """Commit aggregate metrics for a finished episode exactly once.

        Inputs:
            status: Final episode status string.

        Outputs:
            None.

        Behavior:
            Stores per-episode reward, step count, and terminal status for
            downstream evaluation and demo reporting.
        """
        if self._episode_metrics_committed:
            return

        self.episode_reward_history.append(self.current_episode_reward)
        self.episode_step_history.append(self.step_count)
        self.episode_status_history.append(status)
        self._episode_metrics_committed = True

    def get_metrics(self) -> Dict[str, float]:
        """Return aggregate performance metrics for evaluation.

        Inputs:
            None.

        Outputs:
            Dictionary containing episodes, success_rate, avg_reward, and avg_steps.

        Behavior:
            Computes lightweight aggregate metrics from cumulative counters.
        """
        if self.episode_count == 0:
            return {
                "episodes": 0,
                "episode_count": 0,
                "total_reward": 0.0,
                "success_rate": 0.0,
                "avg_reward": 0.0,
                "avg_steps": 0.0,
                "completed_episodes": 0,
            }

        completed_episodes = len(self.episode_status_history)
        avg_episode_reward = (
            sum(self.episode_reward_history) / completed_episodes
            if completed_episodes > 0
            else 0.0
        )
        avg_episode_steps = (
            sum(self.episode_step_history) / completed_episodes
            if completed_episodes > 0
            else 0.0
        )

        return {
            "episodes": self.episode_count,
            "episode_count": self.episode_count,
            "completed_episodes": completed_episodes,
            "success_rate": self.success_count / self.episode_count,
            "total_reward": float(self.total_reward),
            "avg_reward": self.total_reward / self.episode_count,
            "avg_steps": self.total_steps / self.episode_count,
            "avg_episode_reward": avg_episode_reward,
            "avg_episode_steps": avg_episode_steps,
        }

    def export_metrics(self) -> Dict[str, Any]:
        """Export aggregate and historical metrics in JSON-friendly form.

        Inputs:
            None.

        Outputs:
            Dictionary containing metrics, reward history, and step history.

        Behavior:
            Produces a portable evaluation payload suitable for dashboards or judging.
        """
        return {
            "metrics": self.get_metrics(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "success_count": self.success_count,
            "total_reward": self.total_reward,
            "reward_history": list(self.reward_history),
            "episode_reward_history": list(self.episode_reward_history),
            "episode_step_history": list(self.episode_step_history),
            "episode_status_history": list(self.episode_status_history),
            "history": list(self.history),
        }

    def get_demo_trace(self) -> List[Dict[str, Any]]:
        """Return the current episode trace in a demo-friendly structure.

        Inputs:
            None.

        Outputs:
            List of step dictionaries with readable state, action, reward, and error data.

        Behavior:
            Exposes a clean trace for judge demos, UI rendering, or presentation output.
        """
        return list(self.demo_trace)

    def describe_environment(self) -> Dict[str, Any]:
        """Return a compact summary of the environment design and scale.

        Inputs:
            None.

        Outputs:
            Dictionary describing packages, curriculum, and reward configuration.

        Behavior:
            Explains that this is a simplified npm simulation focused on
            decision-making while also surfacing judge-friendly environment
            metadata.
        """
        return {
            "name": "NPMResolverEnv",
            "summary": (
                "Simplified npm dependency-resolution environment for reinforcement "
                "learning and policy evaluation."
            ),
            "focus": (
                "The environment emphasizes sequential decision-making, "
                "dependency reasoning, and measurable repair progress."
            ),
            "compatibility_rule": (
                "Peer dependencies are considered compatible when they match "
                "exactly or share the same major version."
            ),
            "deletion_policy": (
                "Core packages cannot be deleted. Non-core packages can be "
                "removed with an explicit reward penalty."
            ),
            "package_count": len(self.allowed_packages),
            "registry_package_count": len(self.registry),
            "levels": self.get_curriculum_overview(),
            "reward_profile": {
                "step_penalty": self.step_penalty,
                "progress_reward_per_error": self.progress_reward_per_error,
                "regression_penalty_per_error": self.regression_penalty_per_error,
                "no_op_penalty": self.no_op_penalty,
                "invalid_action_penalty": self.invalid_action_penalty,
                "deletion_penalty": self.deletion_penalty,
                "timeout_penalty": self.timeout_penalty,
                "timeout_reward_cap": self.timeout_reward_cap,
                "success_reward": self.success_reward,
                "fatal_penalty": self.fatal_penalty,
            },
        }

    def get_env_info(self) -> Dict[str, Any]:
        """Return a compact interface-level description of the environment.

        Inputs:
            None.

        Outputs:
            Dictionary describing the state space, action space, reward structure,
            and episode horizon.

        Behavior:
            Exposes a clean summary suitable for agents, demos, and hackathon
            reviewers.
        """
        return {
            "name": "NPMResolverEnv",
            "state_space": (
                "JSON dependency map plus deterministic npm-style conflict log "
                "and step counter."
            ),
            "action_space": (
                "Action(package_to_update: str, new_version: semver string or "
                "\"DELETE\" for non-core packages)."
            ),
            "reward_structure": (
                "Fixed step cost, per-error progress reward, per-error regression "
                "penalty, no-op penalty, deletion penalty, success bonus, and "
                "timeout penalty."
            ),
            "max_steps": self.max_steps,
        }

    def render_demo_trace(self) -> str:
        """Render the demo trace as a readable multi-line text report.

        Inputs:
            None.

        Outputs:
            Human-readable string describing the current demo trace.

        Behavior:
            Formats step transitions, rewards, and terminal outcomes for live
            demos or judge walkthroughs.
        """
        if not self.demo_trace:
            return "No demo trace available."

        lines = []
        for entry in self.demo_trace:
            lines.append(entry["summary"])
            lines.append(f"State:\n{entry['state_json']}")
            lines.append(f"Log: {entry['error']}")
        return "\n\n".join(lines)

    def get_curriculum_overview(self) -> Dict[str, Any]:
        """Return a structured summary of curriculum levels and scenario counts.

        Inputs:
            None.

        Outputs:
            Dictionary describing the curriculum hierarchy.

        Behavior:
            Provides a judge-friendly view of environment scalability and level
            progression.
        """
        return {
            level_name: {
                "description": self.level_descriptions.get(level_name, "Custom level"),
                "scenario_count": len(level_scenarios),
            }
            for level_name, level_scenarios in self.levels.items()
        }

    def get_final_state_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the current environment terminal state.

        Inputs:
            None.

        Outputs:
            Dictionary containing status, error counts, and readable state data.

        Behavior:
            Exposes the latest environment outcome in a presentation-friendly
            format for demos or APIs.
        """
        return {
            "status": self.episode_status,
            "resolved": self.last_error_count == 0,
            "remaining_errors": self.last_error_count,
            "steps": self.step_count,
            "step_count": self.step_count,
            "episode_reward": self.current_episode_reward,
            "total_reward": self.total_reward,
            "level": self.active_level,
            "scenario_index": self.current_scenario_index,
            "state": dict(self._get_observable_state()),
            "state_json": self._safe_json_dumps(self._get_observable_state()),
            "error_log": self.last_error_log,
        }

    def print_metrics_summary(self) -> str:
        """Render and print the current aggregate metrics summary.

        Inputs:
            None.

        Outputs:
            JSON-formatted metrics summary string.

        Behavior:
            Produces a readable evaluation snapshot and prints it for demos or
            scripts while also returning the same text.
        """
        metrics_summary = self._safe_json_dumps(self.get_metrics())
        print(metrics_summary)
        return metrics_summary

    def _finalize_transition(
        self,
        error_log: str,
        reward: int,
        done: bool,
        status: str,
        action: Optional[Action] = None,
        errors_remaining: Optional[int] = None,
        old_error_count: Optional[int] = None,
        new_error_count: Optional[int] = None,
    ) -> Tuple[Observation, int, bool, dict]:
        """Commit transition bookkeeping and return a canonical step result.

        Inputs:
            error_log: Final log string for the transition.
            reward: Scalar reward.
            done: Terminal flag.
            status: High-level transition status.
            errors_remaining: Optional explicit remaining error count.
            old_error_count: Optional pre-action error count.
            new_error_count: Optional post-action error count.

        Outputs:
            Tuple of `(observation, reward, done, info)`.

        Behavior:
            Updates episode tracking and guarantees a consistent return shape.
        """
        error_count = (
            self._count_errors(error_log)
            if errors_remaining is None
            else errors_remaining
        )
        self._record_step_metrics(
            action=action,
            reward=reward,
            done=done,
            status=status,
            error_log=error_log,
            old_error_count=old_error_count,
            new_error_count=new_error_count,
        )
        self.last_error_log = error_log
        self.last_error_count = error_count
        self.episode_done = done
        self.episode_status = status
        if done:
            self._record_completed_episode(status)
        observation = self._build_observation(error_log)
        info = self._build_info(
            status=status,
            errors_remaining=error_count,
            old_error_count=old_error_count,
            new_error_count=new_error_count,
        )
        return observation, reward, done, info

    def _advance_level_if_ready(self) -> None:
        """Advance the curriculum after enough successful episodes.

        Inputs:
            None.

        Outputs:
            None.

        Behavior:
            Promotes the active curriculum level when the success threshold is met.
        """
        if self.level_success_streak < self.level_success_threshold:
            return

        current_index = self.level_order.index(self.current_level)
        if current_index >= len(self.level_order) - 1:
            return

        self.current_level = self.level_order[current_index + 1]
        self.level_success_streak = 0
        self._log_debug(f"Advanced curriculum to {self.current_level}.")

    def _select_level_scenario(self, level_name: str) -> Tuple[Dict[str, str], int]:
        """Select the next scenario for a level using deterministic ordering.

        Inputs:
            level_name: Valid curriculum level name.

        Outputs:
            Scenario dictionary and deterministic scenario index for the next episode.

        Behavior:
            Cycles through scenarios deterministically to preserve reproducibility.
        """
        level_scenarios = self.levels[level_name]
        index = self.level_indices.get(level_name, 0) % len(level_scenarios)
        self.level_indices[level_name] = index + 1
        return dict(level_scenarios[index]), index + 1

    def _start_episode(
        self, scenario: Dict[str, str], level_name: str
    ) -> Observation:
        """Initialize a new episode from an explicit scenario.

        Inputs:
            scenario: Dependency state to load for the new episode.
            level_name: Label used for the active episode context.

        Outputs:
            Initial `Observation` for the episode.

        Behavior:
            Resets per-episode tracking, clears demo trace data, audits the
            supplied scenario, and returns the starting observation.
        """
        self.step_count = 0
        self.last_action = None
        self.last_error_count = 0
        self.last_error_log = ""
        self.episode_done = False
        self.episode_status = "in_progress"
        self.current_episode_reward = 0
        self._episode_metrics_committed = False
        self.debug_logs = []
        self.demo_trace = []
        self.episode_count += 1
        self.active_level = level_name
        self.current_state = dict(scenario)
        self.last_error_log = self._safe_current_error_log()
        self.last_error_count = self._count_errors(self.last_error_log)
        self._log_debug(
            f"Environment reset at {self.active_level} with "
            f"{self.last_error_count} error(s)."
        )
        return self._build_observation(self.last_error_log)

    def _validate_level(self, level_name: str) -> str:
        """Validate a requested curriculum level and supply a safe fallback.

        Inputs:
            level_name: Requested level name.

        Outputs:
            Valid level name.

        Behavior:
            Falls back to `level_1` when the requested level is missing or empty.
        """
        if level_name in self.levels and self.levels[level_name]:
            return level_name

        self._log_debug(
            f"Invalid or empty level '{level_name}' requested; falling back to level_1."
        )
        return "level_1"

    def _validate_registry_version(self, action: Action) -> Optional[str]:
        """Validate that registry-backed packages use known concrete versions.

        Inputs:
            action: Candidate action after schema validation.

        Outputs:
            Standardized error string when the version is unsupported, else `None`.

        Behavior:
            Prevents agents from exploiting unknown registry versions that would
            otherwise change error counts without representing a valid package state.
        """
        if action.new_version == "DELETE":
            return None

        package_registry = self.registry.get(action.package_to_update)
        if package_registry is None:
            return None

        if action.new_version not in package_registry:
            return self._format_error("Error", "Unknown package version.")

        return None

    def _extract_major_version(self, version: str) -> Optional[int]:
        """Extract the major version number from a lightweight semver string.

        Inputs:
            version: Version string such as `^18.0.0` or `6.0.0`.

        Outputs:
            Major version integer when parseable, otherwise `None`.

        Behavior:
            Supports the simplified version language used by this environment.
        """
        cleaned_version = version.strip()
        if cleaned_version.startswith("^"):
            cleaned_version = cleaned_version[1:]

        parts = cleaned_version.split(".")
        if not parts:
            return None

        try:
            return int(parts[0])
        except ValueError:
            return None

    def is_compatible(self, required: str, actual: str) -> bool:
        """Evaluate lightweight npm-style compatibility between two versions.

        Inputs:
            required: Required version string from registry metadata.
            actual: Actual version string present in the environment state.

        Outputs:
            Boolean compatibility flag.

        Behavior:
            Treats exact matches as compatible and otherwise falls back to a
            simplified same-major-version rule.
        """
        if required == actual:
            return True

        if required == "DELETE" or actual == "DELETE":
            return False

        required_major = self._extract_major_version(required)
        actual_major = self._extract_major_version(actual)
        if required_major is None or actual_major is None:
            return False

        return required_major == actual_major

    def _compute_transition_reward(
        self,
        old_error_count: int,
        new_error_count: int,
        state_changed: bool,
        deletion_performed: bool,
        success: bool,
        timeout_reached: bool,
    ) -> int:
        """Compute the deterministic reward for a completed state transition.

        Inputs:
            old_error_count: Error count before applying the action.
            new_error_count: Error count after applying the action.
            state_changed: Whether the action produced a new dependency state.
            deletion_performed: Whether the transition deleted a non-core package.
            success: Whether the transition resolved the environment.
            timeout_reached: Whether the episode exhausted its step budget.

        Outputs:
            Scalar reward for the transition.

        Behavior:
            Applies a fixed step cost, rewards genuine progress, penalizes
            regressions and no-op loops, and adds terminal bonuses or penalties.
        """
        reward = self.step_penalty
        error_delta = old_error_count - new_error_count

        if not state_changed:
            reward += self.no_op_penalty
        elif error_delta > 0:
            reward += error_delta * self.progress_reward_per_error
        elif error_delta < 0:
            reward -= abs(error_delta) * self.regression_penalty_per_error
        else:
            reward += self.stalled_progress_penalty

        if deletion_performed:
            reward += self.deletion_penalty

        if success:
            reward += self.success_reward
        elif timeout_reached:
            reward += self.timeout_penalty
            reward = min(reward, self.timeout_reward_cap)

        return reward

    def _finalize_invalid_action(
        self,
        error_log: str,
        action: Optional[Action],
        old_error_count: int,
        timeout_reached: bool,
    ) -> Tuple[Observation, int, bool, dict]:
        """Finalize a rejected action with consistent penalties and timeout handling.

        Inputs:
            error_log: Human-readable reason the action was rejected.
            action: Rejected action, if available.
            old_error_count: Error count before the rejected action.
            timeout_reached: Whether the episode has exhausted its step budget.

        Outputs:
            Canonical `(observation, reward, done, info)` tuple.

        Behavior:
            Applies the invalid-action penalty, upgrades to timeout on the final
            allowed step, and preserves the unresolved environment state.
        """
        reward = self.invalid_action_penalty
        done = True
        status = "failed"
        final_error_log = error_log

        if timeout_reached:
            reward += self.timeout_penalty
            done = True
            status = "timeout"
            final_error_log = self._compose_timeout_log(self._safe_current_error_log())

        return self._finalize_transition(
            error_log=final_error_log,
            reward=reward,
            done=done,
            status=status,
            action=action,
            errors_remaining=old_error_count,
            old_error_count=old_error_count,
            new_error_count=old_error_count,
        )

    def _compose_timeout_log(self, current_error_log: str) -> str:
        """Create a readable timeout log that preserves the true dependency state.

        Inputs:
            current_error_log: Audit log for the unresolved current state.

        Outputs:
            Human-readable timeout message, optionally followed by current errors.

        Behavior:
            Keeps timeout semantics visible while preserving observability.
        """
        if not current_error_log or current_error_log == self.success_message:
            return self.timeout_message

        return f"{self.timeout_message}\n{current_error_log}"

    def _validate_action_instance(self, action: Action) -> Optional[str]:
        """Validate a strict Action object for use in `step()`.

        Inputs:
            action: Candidate `Action` instance.

        Outputs:
            Error string when invalid, otherwise `None`.

        Behavior:
            Enforces the OpenEnv contract while still protecting the environment
            against malformed external inputs.
        """
        if not isinstance(action, Action):
            return self._format_error("Error", "Step input must be an Action.")

        if action.package_to_update is None or action.new_version is None:
            return self._format_error("Error", "Action fields cannot be None.")

        if not isinstance(action.package_to_update, str) or not isinstance(
            action.new_version, str
        ):
            return self._format_error("Error", "Action fields must be strings.")

        if action.package_to_update.strip() == "" or action.new_version.strip() == "":
            return self._format_error("Error", "Action fields cannot be empty.")

        return None

    def _is_valid_version_format(self, version: str) -> bool:
        """Validate supported version string formats.

        Inputs:
            version: Candidate version string.

        Outputs:
            Boolean validity flag.

        Behavior:
            Accepts `^x.x.x`, `x.x.x`, and `DELETE`.
        """
        return self.VERSION_PATTERN.fullmatch(version) is not None

    def _generate_error_log(self, dependencies: Dict[str, str]) -> str:
        """Audit dependency state against the mock registry.

        Inputs:
            dependencies: Package-to-version dependency mapping.

        Outputs:
            Success string or newline-delimited error log.

        Behavior:
            Detects invalid state, empty state, unknown metadata, missing peer
            dependencies, and simplified major-version compatibility conflicts.
        """
        if not self._is_valid_state(dependencies):
            return self._format_error("FATAL", "Invalid dependency state.")

        if not dependencies:
            return self._format_error("Error", "Empty dependency state.")

        errors: List[str] = []

        for package_name, package_version in dependencies.items():
            package_registry = self.registry.get(package_name)
            if package_registry is None:
                continue

            package_metadata = package_registry.get(package_version)
            if package_metadata is None:
                errors.append(
                    self._format_error(
                        "Error",
                        f"Unknown registry metadata for {package_name}@{package_version}.",
                    )
                )
                continue

            required_dependencies = package_metadata.get("requires", {})
            for required_package, required_version in required_dependencies.items():
                current_version = dependencies.get(required_package)

                if current_version is None:
                    errors.append(
                        f"Missing peer dependency: {package_name}@{package_version} "
                        f"requires {required_package}@{required_version}"
                    )
                    continue

                if not self.is_compatible(required_version, current_version):
                    errors.append(
                        f"Conflict: {package_name}@{package_version} requires "
                        f"{required_package}@{required_version}, but found "
                        f"{current_version}"
                    )

        if errors:
            return "\n".join(errors)

        return self.success_message

    def _safe_current_error_log(self) -> str:
        """Generate an audit log for the current environment state.

        Inputs:
            None.

        Outputs:
            Audit string for `self.current_state`.

        Behavior:
            Uses the same safe validation path as normal transitions.
        """
        return self._generate_error_log(self.current_state)

    def _apply_action_to_state(
        self, state: Dict[str, str], action: Action
    ) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """Compute the next state without mutating the current one.

        Inputs:
            state: Source dependency state.
            action: Validated action object.

        Outputs:
            Tuple of `(next_state, error_message)`.

        Behavior:
            Applies anti-cheat rules, allows controlled non-core deletion, and
            returns a new state or a standardized transition error without
            partially mutating the source state.
        """
        next_state = dict(state)

        if action.package_to_update not in next_state:
            return None, self._format_error("Error", "Package not found.")

        if action.new_version == "DELETE":
            if action.package_to_update in self.untouchable_packages:
                return None, self._format_error(
                    "FATAL", "Cannot remove core framework."
                )

            del next_state[action.package_to_update]
        else:
            next_state[action.package_to_update] = action.new_version

        if not self._is_valid_state(next_state) or len(next_state) < 2:
            return None, self._format_error("FATAL", "Invalid dependency state.")

        return next_state, None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full environment state for storage or transport.

        Inputs:
            None.

        Outputs:
            JSON-friendly dictionary snapshot of environment internals.

        Behavior:
            Produces a safe, framework-friendly payload suitable for checkpointing.
        """
        return {
            "current_state": dict(self._get_observable_state()),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "last_action": None if self.last_action is None else self.last_action.to_dict(),
            "last_error_count": self.last_error_count,
            "last_error_log": self.last_error_log,
            "episode_done": self.episode_done,
            "episode_status": self.episode_status,
            "current_scenario_index": self.current_scenario_index,
            "current_episode_reward": self.current_episode_reward,
            "debug": self.debug,
            "debug_logs": list(self.debug_logs),
            "demo_mode": self.demo_mode,
            "demo_trace": list(self.demo_trace),
            "current_level": self.current_level,
            "active_level": self.active_level,
            "demo_scenario": dict(self.demo_scenario),
            "level_descriptions": dict(self.level_descriptions),
            "levels": {
                level_name: [dict(scenario) for scenario in level_scenarios]
                for level_name, level_scenarios in self.levels.items()
            },
            "registry": self._normalize_registry(self.registry),
            "level_indices": dict(self.level_indices),
            "success_count": self.success_count,
            "level_success_streak": self.level_success_streak,
            "level_success_threshold": self.level_success_threshold,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "history": list(self.history),
            "reward_history": list(self.reward_history),
            "episode_reward_history": list(self.episode_reward_history),
            "episode_step_history": list(self.episode_step_history),
            "episode_status_history": list(self.episode_status_history),
            "step_penalty": self.step_penalty,
            "progress_reward_per_error": self.progress_reward_per_error,
            "regression_penalty_per_error": self.regression_penalty_per_error,
            "stalled_progress_penalty": self.stalled_progress_penalty,
            "no_op_penalty": self.no_op_penalty,
            "invalid_action_penalty": self.invalid_action_penalty,
            "deletion_penalty": self.deletion_penalty,
            "timeout_penalty": self.timeout_penalty,
            "timeout_reward_cap": self.timeout_reward_cap,
            "success_reward": self.success_reward,
            "fatal_penalty": self.fatal_penalty,
        }

    def from_dict(self, state_dict: Dict[str, Any]) -> Observation:
        """Safely restore environment state from a serialized dictionary.

        Inputs:
            state_dict: Dictionary previously created by `to_dict()`.

        Outputs:
            Observation representing the restored state.

        Behavior:
            Restores valid fields, repairs invalid input with safe defaults, and
            re-audits the restored dependency state.
        """
        try:
            if not isinstance(state_dict, dict):
                return self.reset()

            raw_registry = state_dict.get("registry")
            if isinstance(raw_registry, str):
                try:
                    raw_registry = json.loads(raw_registry)
                except json.JSONDecodeError:
                    raw_registry = self.registry
            if raw_registry is not None:
                self.registry = self._normalize_registry(raw_registry)

            raw_levels = state_dict.get("levels")
            if raw_levels is not None:
                self.levels = self._normalize_levels(raw_levels)

            raw_demo_scenario = state_dict.get("demo_scenario")
            if raw_demo_scenario is not None and self._is_well_formed_state(
                raw_demo_scenario, require_allowed_packages=False
            ):
                self.demo_scenario = {
                    str(package_name): str(package_version)
                    for package_name, package_version in raw_demo_scenario.items()
                }

            raw_level_descriptions = state_dict.get("level_descriptions", {})
            if isinstance(raw_level_descriptions, dict):
                self.level_descriptions = {
                    str(level_name): str(description)
                    for level_name, description in raw_level_descriptions.items()
                }

            self._refresh_cached_metadata()

            self.step_count = max(0, int(state_dict.get("step_count", 0)))
            self.max_steps = max(1, int(state_dict.get("max_steps", self.max_steps)))
            self.step_penalty = int(state_dict.get("step_penalty", self.step_penalty))
            self.progress_reward_per_error = int(
                state_dict.get(
                    "progress_reward_per_error", self.progress_reward_per_error
                )
            )
            self.regression_penalty_per_error = int(
                state_dict.get(
                    "regression_penalty_per_error",
                    self.regression_penalty_per_error,
                )
            )
            self.stalled_progress_penalty = int(
                state_dict.get(
                    "stalled_progress_penalty", self.stalled_progress_penalty
                )
            )
            self.no_op_penalty = int(
                state_dict.get("no_op_penalty", self.no_op_penalty)
            )
            self.invalid_action_penalty = int(
                state_dict.get(
                    "invalid_action_penalty", self.invalid_action_penalty
                )
            )
            self.deletion_penalty = int(
                state_dict.get("deletion_penalty", self.deletion_penalty)
            )
            self.timeout_penalty = int(
                state_dict.get("timeout_penalty", self.timeout_penalty)
            )
            self.timeout_reward_cap = int(
                state_dict.get("timeout_reward_cap", self.timeout_reward_cap)
            )
            self.success_reward = int(
                state_dict.get("success_reward", self.success_reward)
            )
            self.fatal_penalty = int(
                state_dict.get("fatal_penalty", self.fatal_penalty)
            )
            self.debug = bool(state_dict.get("debug", self.debug))
            raw_logs = state_dict.get("debug_logs", [])
            self.debug_logs = (
                [str(item) for item in raw_logs][-100:]
                if isinstance(raw_logs, list)
                else []
            )
            self.demo_mode = bool(state_dict.get("demo_mode", self.demo_mode))
            raw_demo_trace = state_dict.get("demo_trace", [])
            self.demo_trace = (
                list(raw_demo_trace) if isinstance(raw_demo_trace, list) else []
            )
            self.current_level = self._validate_level(
                str(state_dict.get("current_level", self.current_level))
            )
            self.active_level = self._validate_level(
                str(state_dict.get("active_level", self.current_level))
            )
            raw_level_indices = state_dict.get("level_indices", {})
            if isinstance(raw_level_indices, dict):
                self.level_indices = {
                    level_name: max(0, int(raw_level_indices.get(level_name, 0)))
                    for level_name in self.levels
                }
            else:
                self.level_indices = {level_name: 0 for level_name in self.levels}
            self.success_count = max(0, int(state_dict.get("success_count", 0)))
            self.level_success_streak = max(
                0, int(state_dict.get("level_success_streak", 0))
            )
            self.level_success_threshold = max(
                1,
                int(
                    state_dict.get(
                        "level_success_threshold", self.level_success_threshold
                    )
                ),
            )
            self.total_steps = max(0, int(state_dict.get("total_steps", 0)))
            self.episode_count = max(0, int(state_dict.get("episode_count", 0)))
            self.total_reward = int(state_dict.get("total_reward", 0))
            self.current_scenario_index = max(
                0, int(state_dict.get("current_scenario_index", 0))
            )
            self.current_episode_reward = int(
                state_dict.get("current_episode_reward", 0)
            )
            raw_history = state_dict.get("history", [])
            self.history = list(raw_history) if isinstance(raw_history, list) else []
            raw_reward_history = state_dict.get("reward_history", [])
            self.reward_history = (
                [int(item) for item in raw_reward_history]
                if isinstance(raw_reward_history, list)
                else []
            )
            raw_episode_reward_history = state_dict.get("episode_reward_history", [])
            self.episode_reward_history = (
                [int(item) for item in raw_episode_reward_history]
                if isinstance(raw_episode_reward_history, list)
                else []
            )
            raw_episode_step_history = state_dict.get("episode_step_history", [])
            self.episode_step_history = (
                [int(item) for item in raw_episode_step_history]
                if isinstance(raw_episode_step_history, list)
                else []
            )
            raw_episode_status_history = state_dict.get("episode_status_history", [])
            self.episode_status_history = (
                [str(item) for item in raw_episode_status_history]
                if isinstance(raw_episode_status_history, list)
                else []
            )
            self.episode_done = bool(state_dict.get("episode_done", False))
            self.episode_status = str(state_dict.get("episode_status", "in_progress"))
            self._episode_metrics_committed = self.episode_done

            raw_state = state_dict.get("current_state", {})
            self.current_state = (
                dict(raw_state) if self._is_valid_state(raw_state) else {}
            )

            raw_action = state_dict.get("last_action")
            if isinstance(raw_action, dict):
                try:
                    self.last_action = Action.from_json(raw_action)
                except (TypeError, ValueError):
                    self.last_action = None
            else:
                self.last_action = None

            self.last_error_log = self._safe_current_error_log()
            self.last_error_count = self._count_errors(self.last_error_log)
            return self._build_observation(self.last_error_log)
        except Exception as exc:
            self._log_debug(f"from_dict exception: {exc!r}")
            return self.reset()

    def reset(self) -> Observation:
        """Start a new episode from the current curriculum level.

        Inputs:
            None.

        Outputs:
            Initial `Observation` for the new episode.

        Behavior:
            Clears episode tracking, chooses a scenario from the active level,
            audits the new state, and returns the initial observation.
        """
        try:
            self.current_level = self._validate_level(self.current_level)
            selected_scenario, scenario_index = self._select_level_scenario(
                self.current_level
            )
            self.current_scenario_index = scenario_index

            if not self._is_valid_state(selected_scenario):
                selected_scenario = self.levels["level_1"][0]
                self.current_level = "level_1"
                self.current_scenario_index = 1
                self._log_debug("Selected scenario was invalid; fell back to level_1.")
            return self._start_episode(selected_scenario, self.current_level)
        except Exception as exc:
            self.current_state = {}
            self.last_action = None
            self.last_error_log = self._format_error(
                "FATAL", "Reset failed; environment recovered safely."
            )
            self.last_error_count = 0
            self.episode_done = False
            self.episode_status = "failed"
            self._log_debug(f"Reset exception: {exc!r}")
            return self._build_observation(self.last_error_log)

    def run_demo_episode(self, agent_function: Any) -> List[Dict[str, Any]]:
        """Run a full fixed demo episode using an external agent callback.

        Inputs:
            agent_function: Callable accepting an `Observation` and returning an `Action`.

        Outputs:
            Demo trace list for the full episode.

        Behavior:
            Loads a fixed solvable scenario, steps until success or termination,
            and returns a readable structured trace for live demos or UIs.
        """
        if not callable(agent_function):
            raise TypeError("agent_function must be callable.")

        self.current_scenario_index = 1
        observation = self._start_episode(self.demo_scenario, "demo")

        while not self.episode_done:
            proposed_action = agent_function(observation)
            if isinstance(proposed_action, dict):
                try:
                    action = Action.from_json(proposed_action)
                except (TypeError, ValueError):
                    action = proposed_action  # type: ignore[assignment]
            else:
                action = proposed_action

            observation, _, done, _ = self.step(action)  # type: ignore[arg-type]
            if done:
                break

        return self.get_demo_trace()

    def step(self, action: Action) -> Tuple[Observation, int, bool, dict]:
        """Advance the environment by one step using a strict Action input.

        Inputs:
            action: `Action` instance describing the desired dependency change.

        Outputs:
            Tuple of `(observation, reward, done, info)`.

        Behavior:
            Applies timeout rules, validates input, enforces anti-cheat logic,
            performs an immutable transition, computes reward, checks success, and
            always returns a stable OpenEnv-compatible result.
        """
        state_snapshot = (
            dict(self.current_state) if self._is_valid_state(self.current_state) else {}
        )
        last_error_log_snapshot = self.last_error_log
        last_error_count_snapshot = self.last_error_count
        success_count_snapshot = self.success_count
        level_success_streak_snapshot = self.level_success_streak
        current_level_snapshot = self.current_level
        active_level_snapshot = self.active_level
        episode_done_snapshot = self.episode_done
        episode_status_snapshot = self.episode_status
        last_action_snapshot = self.last_action

        try:
            if self.episode_done:
                observation = self._build_observation(self.last_error_log)
                info = self._build_info(
                    status=self.episode_status,
                    errors_remaining=self.last_error_count,
                )
                return observation, 0, True, info

            self.step_count += 1
            old_error_count = self.last_error_count

            timeout_reached = self.step_count >= self.max_steps

            if not self._is_valid_state(self.current_state):
                self.current_state = {}
                self.last_action = None
                self._log_debug("Detected invalid current_state before applying action.")
                return self._finalize_transition(
                    error_log=self._format_error("FATAL", "Invalid dependency state."),
                    reward=self.fatal_penalty,
                    done=True,
                    status="failed",
                    action=None,
                    errors_remaining=0,
                    old_error_count=old_error_count,
                    new_error_count=0,
                )

            validation_error = self._validate_action_instance(action)
            if validation_error is not None:
                self.last_action = None
                self._log_debug(f"Rejected action during validation: {validation_error}")
                return self._finalize_invalid_action(
                    error_log=validation_error,
                    action=None,
                    old_error_count=old_error_count,
                    timeout_reached=timeout_reached,
                )

            self.last_action = Action(
                package_to_update=action.package_to_update.strip(),
                new_version=action.new_version.strip(),
            )

            if self.last_action.package_to_update not in self.current_state:
                self._log_debug(
                    f"Rejected unknown package '{self.last_action.package_to_update}'."
                )
                return self._finalize_invalid_action(
                    error_log=self._format_error("Error", "Package not found."),
                    action=self.last_action,
                    old_error_count=old_error_count,
                    timeout_reached=timeout_reached,
                )

            if not self._is_valid_version_format(self.last_action.new_version):
                self._log_debug(
                    f"Rejected invalid version '{self.last_action.new_version}'."
                )
                return self._finalize_invalid_action(
                    error_log=self._format_error("Error", "Invalid version format."),
                    action=self.last_action,
                    old_error_count=old_error_count,
                    timeout_reached=timeout_reached,
                )

            registry_error = self._validate_registry_version(self.last_action)
            if registry_error is not None:
                self._log_debug(
                    f"Rejected unknown version '{self.last_action.new_version}' for "
                    f"package '{self.last_action.package_to_update}'."
                )
                return self._finalize_invalid_action(
                    error_log=registry_error,
                    action=self.last_action,
                    old_error_count=old_error_count,
                    timeout_reached=timeout_reached,
                )

            next_state, transition_error = self._apply_action_to_state(
                self.current_state, self.last_action
            )
            if transition_error is not None:
                self._log_debug(f"Transition blocked: {transition_error}")
                fatal = transition_error.startswith("FATAL:")
                if fatal:
                    return self._finalize_transition(
                        error_log=transition_error,
                        reward=self.fatal_penalty,
                        done=True,
                        status="failed",
                        action=self.last_action,
                        errors_remaining=old_error_count,
                        old_error_count=old_error_count,
                        new_error_count=old_error_count,
                    )

                return self._finalize_invalid_action(
                    error_log=transition_error,
                    action=self.last_action,
                    old_error_count=old_error_count,
                    timeout_reached=timeout_reached,
                )

            self.current_state = next_state or {}
            error_log = self._safe_current_error_log()
            new_error_count = self._count_errors(error_log)
            state_changed = next_state != state_snapshot
            deletion_performed = self.last_action.new_version == "DELETE"

            done = False
            status = "in_progress"
            success = new_error_count == 0

            reward = self._compute_transition_reward(
                old_error_count=old_error_count,
                new_error_count=new_error_count,
                state_changed=state_changed,
                deletion_performed=deletion_performed,
                success=success,
                timeout_reached=timeout_reached and not success,
            )

            if success:
                done = True
                status = "success"
                self.success_count += 1
                self.level_success_streak += 1
                self._advance_level_if_ready()
            elif timeout_reached:
                done = True
                status = "timeout"
                error_log = self._compose_timeout_log(error_log)

            self._log_debug(
                f"Applied action {self.last_action} with reward {reward}; "
                f"errors {old_error_count} -> {new_error_count}."
            )
            return self._finalize_transition(
                error_log=error_log,
                reward=reward,
                done=done,
                status=status,
                action=self.last_action,
                errors_remaining=new_error_count,
                old_error_count=old_error_count,
                new_error_count=new_error_count,
            )
        except Exception as exc:
            self.current_state = state_snapshot
            self.last_error_log = last_error_log_snapshot
            self.last_error_count = last_error_count_snapshot
            self.success_count = success_count_snapshot
            self.level_success_streak = level_success_streak_snapshot
            self.current_level = current_level_snapshot
            self.active_level = active_level_snapshot
            self.episode_done = episode_done_snapshot
            self.episode_status = episode_status_snapshot
            self.last_action = last_action_snapshot
            self._log_debug(f"Step exception recovered safely: {exc!r}")
            return self._finalize_transition(
                error_log=self._format_error(
                    "FATAL", "Internal environment failure recovered safely."
                ),
                reward=self.fatal_penalty,
                done=True,
                status="failed",
                action=self.last_action,
                errors_remaining=self.last_error_count,
                old_error_count=self.last_error_count,
                new_error_count=self.last_error_count,
            )

    def register_package_versions(
        self, package_name: str, versions: Dict[str, Dict[str, Dict[str, str]]]
    ) -> None:
        """Register or replace versions for a package in the mock registry.

        Inputs:
            package_name: Package name to register.
            versions: Version metadata map keyed by version string.

        Outputs:
            None.

        Behavior:
            Validates the schema, updates the registry, and refreshes cached
            metadata so new packages immediately participate in the environment.
        """
        if not isinstance(package_name, str) or package_name.strip() == "":
            raise ValueError("package_name must be a non-empty string.")
        if not isinstance(versions, dict) or not versions:
            raise ValueError("versions must be a non-empty dictionary.")

        normalized_versions: Dict[str, Dict[str, Dict[str, str]]] = {}
        for version_name, metadata in versions.items():
            if not isinstance(version_name, str) or version_name.strip() == "":
                raise ValueError("Each version name must be a non-empty string.")
            if not isinstance(metadata, dict):
                raise ValueError("Each version metadata entry must be a dictionary.")

            requires = metadata.get("requires", {})
            if not isinstance(requires, dict):
                raise ValueError("Each version metadata requires field must be a dictionary.")

            normalized_requires: Dict[str, str] = {}
            for dependency_name, dependency_version in requires.items():
                if not isinstance(dependency_name, str) or dependency_name.strip() == "":
                    raise ValueError("Dependency names must be non-empty strings.")
                if not isinstance(dependency_version, str) or dependency_version.strip() == "":
                    raise ValueError("Dependency versions must be non-empty strings.")
                normalized_requires[dependency_name] = dependency_version

            normalized_versions[version_name] = {"requires": normalized_requires}

        self.registry[package_name.strip()] = normalized_versions
        self._refresh_cached_metadata()

    def add_curriculum_scenario(self, level_name: str, scenario: Dict[str, str]) -> None:
        """Add a new deterministic scenario to a curriculum level.

        Inputs:
            level_name: Target curriculum level.
            scenario: Package version mapping for the scenario.

        Outputs:
            None.

        Behavior:
            Validates the scenario and appends it to the requested level so the
            environment can scale to new dependency challenges.
        """
        if not isinstance(level_name, str) or level_name.strip() == "":
            raise ValueError("level_name must be a non-empty string.")

        validated_level = level_name.strip()
        if validated_level not in self.levels:
            self.levels[validated_level] = []
            self.level_descriptions.setdefault(
                validated_level, "Custom curriculum level."
            )
        if not self._is_well_formed_state(scenario, require_allowed_packages=False):
            raise ValueError("scenario must be a valid non-empty package map.")

        self.levels[validated_level].append(dict(scenario))
        self._refresh_cached_metadata()


def _run_interface_validation() -> None:
    """Run silent interface checks for OpenEnv-style external integration.

    Inputs:
        None.

    Outputs:
        None.

    Behavior:
        Uses assertions to validate reset/step signatures, serialization helpers,
        and safe restoration without emitting debug prints.
    """
    env = NPMResolverEnv()
    env.current_level = "level_2"
    initial_observation = env.reset()
    assert isinstance(initial_observation, Observation)
    assert isinstance(initial_observation.to_dict(), dict)

    agent_payload = {
        "package_to_update": "react",
        "new_version": "^18.0.0",
    }
    action = Action.from_json(agent_payload)
    observation, reward, done, info = env.step(action)
    assert isinstance(observation, Observation)
    assert isinstance(reward, int)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    snapshot = env.to_dict()
    restored_env = NPMResolverEnv()
    restored_observation = restored_env.from_dict(snapshot)
    assert isinstance(restored_observation, Observation)
    assert restored_env.to_dict()["current_state"] == snapshot["current_state"]

    invalid_observation, invalid_reward, invalid_done, invalid_info = restored_env.step(  # type: ignore[arg-type]
        "invalid-action"
    )
    assert isinstance(invalid_observation, Observation)
    assert invalid_reward == -5
    assert isinstance(invalid_done, bool)
    assert isinstance(invalid_info, dict)

    noop_env = NPMResolverEnv()
    noop_env.current_level = "level_1"
    noop_env.reset()
    _, noop_reward, noop_done, _ = noop_env.step(
        Action(package_to_update="react", new_version="^17.0.0")
    )
    assert noop_reward < 0
    assert not noop_done

    delete_env = NPMResolverEnv()
    delete_env.current_level = "level_2"
    delete_env.reset()
    _, delete_reward, delete_done, delete_info = delete_env.step(
        Action(package_to_update="react-router-dom", new_version="DELETE")
    )
    assert delete_reward > 0
    assert delete_done
    assert delete_info["status"] == "success"

    regression_env = NPMResolverEnv()
    regression_env.current_level = "level_3"
    regression_env.reset()
    _, progress_reward, progress_done, _ = regression_env.step(
        Action(package_to_update="react-router-dom", new_version="6.0.0")
    )
    assert progress_reward > 0
    assert not progress_done
    _, regression_reward, regression_done, _ = regression_env.step(
        Action(package_to_update="react-router-dom", new_version="5.0.0")
    )
    assert regression_reward < 0
    assert not regression_done

    timeout_env = NPMResolverEnv()
    timeout_env.current_level = "level_3"
    timeout_env.reset()
    timeout_env.max_steps = 1
    _, timeout_reward, timeout_done, timeout_info = timeout_env.step(
        Action(package_to_update="react-router-dom", new_version="6.0.0")
    )
    assert timeout_reward < 0
    assert timeout_done
    assert timeout_info["status"] == "timeout"

    compatibility_env = NPMResolverEnv()
    assert compatibility_env.is_compatible("^18.0.0", "^18.2.0")
    assert compatibility_env.is_compatible("^18.0.0", "18.1.0")
    assert not compatibility_env.is_compatible("^18.0.0", "^17.9.0")


def _run_metrics_validation() -> None:
    """Run a small multi-episode simulation and print metrics for evaluation.

    Inputs:
        None.

    Outputs:
        None.

    Behavior:
        Exercises success, progression, and timeout paths before printing a
        compact metrics snapshot for manual verification.
    """
    env = NPMResolverEnv()
    env.current_level = "level_1"

    env.reset()
    env.step(Action(package_to_update="react", new_version="^18.0.0"))

    env.current_level = "level_2"
    env.reset()
    env.step(Action(package_to_update="react", new_version="^18.0.0"))
    env.step(Action(package_to_update="react-dom", new_version="^18.0.0"))

    env.current_level = "level_3"
    env.reset()
    env.max_steps = 1
    env.step(Action(package_to_update="react-router-dom", new_version="6.0.0"))

    metrics = env.get_metrics()
    exported = env.export_metrics()
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(exported["metrics"], indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "history_entries": len(exported["history"]),
                "reward_points": len(exported["reward_history"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_demo_validation() -> None:
    """Run a simple fixed demo episode and print the structured trace.

    Inputs:
        None.

    Outputs:
        None.

    Behavior:
        Executes a deterministic demo agent, then prints the environment
        description, readable demo trace, and final state summary.
    """

    def demo_agent(observation: Observation) -> Action:
        state = json.loads(observation.current_package_json)
        if state.get("react") != "^18.0.0":
            return Action(package_to_update="react", new_version="^18.0.0")
        return Action(package_to_update="react-dom", new_version="^18.0.0")

    env = NPMResolverEnv()
    trace = env.run_demo_episode(demo_agent)
    print(json.dumps(env.describe_environment(), indent=2, sort_keys=True))
    print(json.dumps(trace, indent=2, sort_keys=True))
    print(json.dumps(env.get_final_state_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _run_interface_validation()
    _run_metrics_validation()
    _run_demo_validation()
