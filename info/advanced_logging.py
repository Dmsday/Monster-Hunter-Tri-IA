"""
Advanced logging system for Monster Hunter RL training.

## Architecture multi-agent
───────────────────────────
Chaque TrainingLogger (un par agent) crée ses propres fichiers dans agent_N/.
Ses handlers de fichiers sont AUSSI enregistrés sur tous les loggers mh_* via
register_file_handler() dans module_logger.py.

Pourquoi PAS de AgentContextFilter sur les handlers de fichier ?
──────────────────────────────────────────────────────────────────
DummyVecEnv exécute tous les sous-envs dans le MÊME thread, séquentiellement.
Quand `env.step(actions)` est appelé, le dernier AgentContext défini dans la
boucle précédente est encore actif — donc tous les logs mh_env/reward_calc
vont dans un seul fichier si on filtre par agent.

La solution retenue : chaque agent reçoit TOUS les logs des modules (contenu
identique entre agents). La séparation stricte par agent est assurée pour les
données d'entraînement (JSONL) via `record_step()`, qui est appelé
explicitement par agent dans MultiAgentTrainer.

Résultat final:
    logs/<exp>/<ts>/
        agent_0/
            console.log           ← WARNING+ via log_info/log_warning (+ tous mh_* logs)
            debug.log             ← DEBUG+ (même contenu que agent_1/debug.log)
            errors.log            ← ERROR+ avec tracebacks
            session_summary.json  ← stats de cet agent
            training_data.jsonl   ← steps/épisodes DE CET AGENT UNIQUEMENT
        agent_1/
            ...
"""

import os
import atexit
import json
import logging
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback

from info.module_logger import get_module_logger, register_file_handler, _GLOBAL_LOG_LEVEL as _INITIAL_GLOBAL_LOG_LEVEL, AgentAwareFormatter

_internal_logger = get_module_logger('advanced_logging')

import gzip
import shutil
from logging.handlers import RotatingFileHandler


class _CompressedRotatingFileHandler(RotatingFileHandler):
    """
    RotatingFileHandler that gzip-compresses rolled-over files.
    Keeps maxBytes per active log; older rotations are stored as .gz.
    """
    def doRollover(self) -> None:
        super().doRollover()
        # Compress all numbered backups that are not yet compressed
        for i in range(1, self.backupCount + 1):
            rolled = f"{self.baseFilename}.{i}"
            if os.path.exists(rolled):
                gz_path = f"{rolled}.gz"
                try:
                    with open(rolled, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(rolled)
                except OSError:
                    pass  # Non-critical: leave uncompressed if gz write fails

# =========================================================================
#  TRAINING LOGGER
# =========================================================================
class AgentContextFilter(logging.Filter):
    """
    Filters log records to only allow those matching a specific agent_id.
    Used to route per-agent file handlers correctly in multi-agent mode.
    Since DummyVecEnv is sequential (one agent active at a time),
    filtering by current AgentContext is safe and accurate.
    """

    def __init__(self, agent_id: int):
        super().__init__()
        self.agent_id = agent_id

    def filter(self, record) -> bool:
        from info.agent_context import AgentContext
        current = AgentContext.get_current_agent()
        # Allow: correct agent, OR no context set (startup/shutdown logs go to agent 0)
        if current is None:
            return self.agent_id == 0
        return current == self.agent_id


class _EnvContextFilter(logging.Filter):
    """
    Filters log records to only allow those matching a specific (agent_id, env_id) pair.
    Used exclusively on per-env console_envX.log handlers.
    Fallback: no context set → only agent 0 / env 0 receives startup logs.
    If only agent context is set (no env yet) → passes through for the right agent.
    """

    def __init__(self, agent_id: int, env_id: int):
        super().__init__()
        self.agent_id = agent_id
        self.env_id   = env_id

    def filter(self, record) -> bool:
        from info.agent_context import AgentContext, EnvContext
        current_agent = AgentContext.get_current_agent()
        current_env   = EnvContext.get_current_env()

        # No context at all → only agent 0 / env 0
        if current_agent is None and current_env is None:
            return self.agent_id == 0 and self.env_id == 0

        agent_ok = (current_agent == self.agent_id) if current_agent is not None else (self.agent_id == 0)

        # Env context not yet set → let it through for the right agent
        # (covers early startup logs before EnvContext is initialised)
        if current_env is None:
            return agent_ok and self.env_id == 0

        return agent_ok and (current_env == self.env_id)
class TrainingLogger:
    """
    Logger per-agent avec fichiers séparés.

    Utilisation :
        # Créer un logger par agent
        tl = TrainingLogger('test1', agent_id=0, num_agents=2)

        # Logguer des données d'entraînement (step data)
        tl.log_step_data({'reward': 0.5, 'hp': 75})

        # Logguer des messages (via API)
        tl.log_info("Message important")

        # Fermer proprement à la fin
        tl.close()
    """

    def __init__(
            self,
            experiment_name: str,
            base_dir: str = "./logs",
            console_log_level: str = "WARNING",
            agent_id: Optional[int] = None,
            instance_id: Optional[int] = None,
            num_agents: int = 1,
            session_timestamp: Optional[str] = None,
    ):
        """
        Args:
            experiment_name:   Nom du run (= nom du dossier).
            base_dir:          Dossier racine pour tous les logs.
            console_log_level: Niveau minimum pour console.log.
            agent_id:          Si défini, logs dans agent_{N}/. Si None = single-agent.
            num_agents:        Nombre total d'agents (informatif).
        """
        self.experiment_name = experiment_name
        self.session_start = datetime.now()
        self.console_log_level = console_log_level
        self.agent_id = agent_id
        self.num_agents = num_agents

        # ── Folder ───────────────────────────────────────────────────────
        timestamp = session_timestamp or self.session_start.strftime("%Y%m%d_%H%M%S")
        base_exp_dir = Path(base_dir) / experiment_name / timestamp

        self._instance_id = instance_id

        if agent_id is not None and instance_id is not None:
            # Per-env files go in agent_N/env_M/
            # Agent-level files (errors.log, session_summary.json) stay in agent_N/
            self.log_dir   = base_exp_dir / f"agent_{agent_id}" / f"env_{instance_id}"
            self.summary_dir = base_exp_dir / f"agent_{agent_id}"
            self.agent_prefix = f"Agent {agent_id} | Env {instance_id} | "
        elif agent_id is not None:
            self.log_dir   = base_exp_dir / f"agent_{agent_id}"
            self.summary_dir = self.log_dir
            self.agent_prefix = f"Agent {agent_id} | "
        else:
            self.log_dir   = base_exp_dir
            self.summary_dir = self.log_dir
            self.agent_prefix = ""

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        # ── Counters ─────────────────────────────────────────────────────
        self.step_count    = 0
        self.episode_count = 0
        self.error_count   = 0
        self.warning_count = 0

        # ── Historique par épisode (pour session_summary) ─────────────────
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int]   = []
        self.episode_hits:    List[int]   = []
        self.episode_deaths:  List[int]   = []
        self.episode_zones:   List[int]   = []

        # ── Session metadata ─────────────────────────────────────────
        self.session_data: Dict[str, Any] = {
            'experiment_name':  experiment_name,
            'agent_id':         agent_id,
            'num_agents':       num_agents,
            'start_time':       self.session_start.isoformat(),
            'end_time':         None,
            'duration_seconds': 0.0,
            'total_steps':      0,
            'total_episodes':   0,
            'errors':           0,
            'warnings':         0,
            'config':           {},
            'episode_statistics': {},
        }

        # ── File loggers ────────────────────────────────────────────
        self._file_handlers: List[logging.Handler] = []
        self._setup_loggers()

        # ── JSONL file (step/episode data per agent) ─────────────────
        self.training_data_file = open(
            self.log_dir / "training_data.jsonl", 'w', encoding='utf-8'
        )

        atexit.register(self._emergency_close)
        _internal_logger.info(f"TrainingLogger prêt : {self.log_dir}")

    # ──────────────────────────────────────────────────────────────────────
    #  SETUP
    # ──────────────────────────────────────────────────────────────────────

    def _setup_loggers(self):
        """
        Creates the three file handlers (console, debug, errors).

        Filtering strategy:
          - When agent_id is set: AgentContextFilter(agent_id) is applied to
            console.log and errors.log before register_file_handler(). Each
            agent_N/ folder only receives logs produced while that agent is
            the active context.
          - reward_debug.log is NOT registered globally — it only receives
            explicit calls via log_reward_step() / log_step_data().
        """
        # Use the level passed by the caller (reflects --log-level CLI arg).
        # _INITIAL_GLOBAL_LOG_LEVEL is captured at import time and is always
        # WARNING; using self.console_log_level fixes the DEBUG-not-written bug.
        console_level = getattr(logging, self.console_log_level.upper(), logging.WARNING)

        # nom unique par combinaison agent+env
        if self.agent_id is not None and self._instance_id is not None:
            name_prefix = f'{self.experiment_name}_agent{self.agent_id}_env{self._instance_id}'
        elif self.agent_id is not None:
            name_prefix = f'{self.experiment_name}_agent{self.agent_id}'
        else:
            name_prefix = self.experiment_name

        # Agent-level filter for errors.log (shared across envs)
        _agent_filter = AgentContextFilter(self.agent_id) if self.agent_id is not None else None
        # Per-env filter for console.log — now that console.log lives in env_N/ subfolder,
        # it must use _EnvContextFilter so each file only receives its own env's logs.
        # Logs without EnvContext (update phase, weight sync) fall to env 0 of the agent,
        # which is acceptable: those are agent-level events, not env-specific.
        if self.agent_id is not None and self._instance_id is not None:
            _console_filter = _EnvContextFilter(self.agent_id, self._instance_id)
        elif self.agent_id is not None:
            _console_filter = _agent_filter
        else:
            _console_filter = None

        base_fmt = AgentAwareFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(agent_prefix)s%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # ── console.log  (one per agent) ──────────────────────────────────
        # The FILE always captures DEBUG+ so nothing is missed in logs.
        # console_level (from --log-level) only controls stdout output, not files.
        self.console_logger = _make_isolated_logger(
            f'{name_prefix}_console', logging.DEBUG
        )
        # console.log inside the env subfolder — 5 MB, 3 backups
        ch = _CompressedRotatingFileHandler(
            self.log_dir / "console.log", encoding='utf-8',
            maxBytes=5 * 1024 * 1024, backupCount=3
        )
        ch.setLevel(logging.DEBUG)  # always capture everything in the file
        ch.setFormatter(base_fmt)
        if _console_filter:
            ch.addFilter(_console_filter)
        self.console_logger.addHandler(ch)
        self._file_handlers.append(ch)
        register_file_handler(ch)

        # ── errors.log  (ERROR+ with traceback) ───────────────────────────
        self.error_logger = _make_isolated_logger(
            f'{name_prefix}_errors', logging.ERROR
        )
        # errors.log : 2 MB, 2 backups
        eh = _CompressedRotatingFileHandler(
            self.summary_dir / "errors.log", encoding='utf-8',
            maxBytes=2 * 1024 * 1024, backupCount=2
        )
        eh.setLevel(logging.ERROR)
        eh.setFormatter(SafeFormatter(
            f'%(asctime)s | STEP:%(step)s | %(levelname)s | {self.agent_prefix}%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        if _agent_filter:
            eh.addFilter(_agent_filter)
        self.error_logger.addHandler(eh)
        self._file_handlers.append(eh)
        register_file_handler(eh)

        # ── reward_debug.log (DEBUG+ verbose) ─────────────────────────────
        # NOT registered globally — receives only explicit calls
        # (log_reward_step, log_step_data).
        self.debug_logger = _make_isolated_logger(
            f'{name_prefix}_debug', logging.DEBUG
        )
        # reward_debug.log : 20 MB, 5 backups (most verbose file)
        dh = _CompressedRotatingFileHandler(
            self.log_dir / "reward_debug.log", encoding='utf-8',
            maxBytes=20 * 1024 * 1024, backupCount=5
        )
        dh.setLevel(logging.DEBUG)
        dh.setFormatter(SafeFormatter(
            f'%(asctime)s | STEP:%(step)s EP:%(episode)s | %(levelname)s | '
            f'{self.agent_prefix}%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.debug_logger.addHandler(dh)
        self._file_handlers.append(dh)
        # register_file_handler(dh) intentionally NOT called

        _internal_logger.debug(f"Handlers enregistrés pour {name_prefix}")
        # Prevent records from propagating to root logger (would duplicate in stdout)
        self.error_logger.propagate = False
        self.debug_logger.propagate = False
        self.console_logger.propagate = False

    # ──────────────────────────────────────────────────────────────────────
    #  EMERGENCY CLOSURE (atexit)
    # ──────────────────────────────────────────────────────────────────────

    def _emergency_close(self):
        """Flush/close all files if close() was never called (atexit safety net)."""
        try:
            # Write session summary if not already done
            summary_path = self.summary_dir / "session_summary.json"
            if not summary_path.exists():
                try:
                    self.save_session_summary()
                except Exception:
                    pass

            if hasattr(self, 'training_data_file') and not self.training_data_file.closed:
                self.training_data_file.flush()
                self.training_data_file.close()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    #  LOG API
    # ──────────────────────────────────────────────────────────────────────

    def log_error(self, error: Exception, context: str = ""):
        self.error_count += 1
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        tb_str = "".join(tb_lines).strip()

        # Build the error message with clear separator for readability
        separator = "=" * 70
        agent_env = ""
        if self.agent_id is not None:
            agent_env = f"Agent {self.agent_id}"
            if self._instance_id is not None:
                agent_env += f" | Env {self._instance_id}"
            agent_env = f" [{agent_env}]"

        msg_parts = [
            separator,
            f"ERROR #{self.error_count}{agent_env}",
            f"Context: {context}" if context else None,
            f"Exception: {type(error).__name__}: {error}",
            separator[:35],
        ]
        if tb_str:
            msg_parts.append(f"Traceback:\n{tb_str}")
        msg_parts.append(separator)
        msg_parts.append("")  # blank line after

        msg = "\n".join(p for p in msg_parts if p is not None)

        self.error_logger.error(msg, extra={'step': self.step_count})
        # Also write full error to console.log for complete history
        self.console_logger.error(msg)
        self.debug_logger.error(
            f"ERROR [{context}]: {type(error).__name__}: {error} — see errors.log for full traceback",
            extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def log_warning(self, message: str):
        self.warning_count += 1
        self.console_logger.warning(message)
        self.debug_logger.warning(
            message, extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def log_info(self, message: str):
        self.console_logger.info(message)
        self.debug_logger.info(
            message, extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def log_debug(self, message: str):
        self.debug_logger.debug(
            message, extra={'step': self.step_count, 'episode': self.episode_count}
        )

    def log_reward_step(
            self,
            reward: float,
            breakdown: dict,
            step: int = None,
            episode: int = None,
            breakdown_detailed: dict = None,
    ):
        """Write full per-step reward breakdown to debug.log (all categories + subcategories).
        Called every step from LoggingCallback._process_step().
        """
        s = step if step is not None else self.step_count
        e = episode if episode is not None else self.episode_count

        if not breakdown:
            self.debug_logger.debug(
                f"[EP:{e:>4} STEP:{s:>7}] reward={reward:+.4f} | no breakdown available",
                extra={'step': s, 'episode': e},
            )
            return

        # Top-level categories (all, even zero — full picture every step)
        top_parts = "  ".join(
            f"{cat}={val:+.4f}"
            for cat, val in sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Subcategories (only non-zero to keep readable)
        detail_parts = ""
        if breakdown_detailed:
            detail_parts = "  ".join(
                f"{subcat}={val:+.4f}"
                for subcat, val in sorted(breakdown_detailed.items(), key=lambda x: abs(x[1]), reverse=True)
                if val != 0.0
            )

        line = f"[EP:{e:>4} STEP:{s:>7}] total={reward:+.4f} || {top_parts}"
        if detail_parts:
            line += f"\n                              └─ details: {detail_parts}"

        self.debug_logger.debug(line, extra={'step': s, 'episode': e})

    def log_episode_reward_summary(
            self,
            episode: int,
            total_reward: float,
            breakdown_accumulator: dict,
    ):
        """Write episode-level reward summary to debug.log on reset.
        breakdown_accumulator: summed per-category rewards over the episode.
        """
        sep = "-" * 70
        lines = [
            sep,
            f"EPISODE {episode:>5} RESET  |  total_reward={total_reward:+.4f}",
        ]
        if breakdown_accumulator:
            lines.append("  Cumulative breakdown:")
            for cat, val in sorted(breakdown_accumulator.items(), key=lambda x: abs(x[1]), reverse=True):
                lines.append(f"    {cat:<30} {val:+.4f}")
        lines.append(sep)
        for line in lines:
            self.debug_logger.debug(line, extra={'step': self.step_count, 'episode': episode})

    # ──────────────────────────────────────────────────────────────────────
    #  JSONL — DONNÉES D'ENTRAÎNEMENT (per-agent via record_step)
    # ──────────────────────────────────────────────────────────────────────

    def log_step_data(self, data: Dict[str, Any]):
        """Ajoute un enregistrement de step dans training_data.jsonl."""
        self.step_count += 1
        entry = {
            'timestamp':         datetime.now().isoformat(),
            'real_time_elapsed': (datetime.now() - self.session_start).total_seconds(),
            'step':              self.step_count,
            'episode':           self.episode_count,
            **data,
        }
        self.training_data_file.write(json.dumps(entry) + '\n')
        self.training_data_file.flush()

        if self.step_count % 100 == 0:
            self.debug_logger.debug(
                f"Step {self.step_count} | reward: {data.get('reward', 0):.2f} | "
                f"HP: {data.get('hp', 0)}",
                extra={'step': self.step_count, 'episode': self.episode_count}
            )

    def log_episode_end(self, episode_data: Dict[str, Any]):
        """Log fin d'épisode dans les fichiers et dans le JSONL."""
        self.episode_count += 1

        self.episode_rewards.append(float(episode_data.get('total_reward', 0.0)))
        self.episode_lengths.append(int(episode_data.get('length', 0)))
        self.episode_hits.append(int(episode_data.get('hits', 0)))
        self.episode_deaths.append(int(episode_data.get('deaths', 0)))
        self.episode_zones.append(int(episode_data.get('zones_discovered', 0)))

        self.console_logger.info(
            f"Episode {self.episode_count} END | "
            f"reward: {episode_data.get('total_reward', 0):.2f} | "
            f"length: {episode_data.get('length', 0)}"
        )
        self.debug_logger.info(
            f"Episode END | reward: {episode_data.get('total_reward', 0):.2f} | "
            f"length: {episode_data.get('length', 0)} | "
            f"hits: {episode_data.get('hits', 0)} | "
            f"deaths: {episode_data.get('deaths', 0)}",
            extra={'step': self.step_count, 'episode': self.episode_count}
        )
        self.training_data_file.write(
            json.dumps({'timestamp': datetime.now().isoformat(),
                        'type': 'EPISODE_END',
                        'episode': self.episode_count,
                        **episode_data}) + '\n'
        )
        self.training_data_file.flush()

    def log_config(self, config: Dict[str, Any]):
        self.session_data['config'] = config
        if 'command' in config:
            self.session_data['command'] = config['command']
        _env_header = f"Env {self._instance_id}" if self._instance_id is not None else "—"
        self.debug_logger.info(
            f"[Agent {self.agent_id} | {_env_header}] Training config:\n{json.dumps(config, indent=2)}",
            extra={'step': 0, 'episode': 0}
        )
        # Write session header as first line of training_data.jsonl
        header = {
            'type': 'SESSION_START',
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
            'agent_id': self.agent_id,
            'instance_id': self._instance_id,
            'command': config.get('command', ''),
            'num_agents': self.num_agents,
        }
        self.training_data_file.write(json.dumps(header) + '\n')
        self.training_data_file.flush()

    def log_checkpoint(self, checkpoint_path: str, timesteps: int):
        self.console_logger.info(
            f"Checkpoint saved: {checkpoint_path} ({timesteps:,} steps)"
        )
        self.debug_logger.info(
            f"Checkpoint saved: {checkpoint_path} ({timesteps:,} steps)",
            extra={'step': self.step_count, 'episode': self.episode_count}
        )

    # ──────────────────────────────────────────────────────────────────────
    #  RÉSUMÉ DE SESSION & FERMETURE
    # ──────────────────────────────────────────────────────────────────────

    def save_session_summary(self):
        """Write session_summary.json with 'command' always first for easy copy-paste."""
        now = datetime.now()
        self.session_data['end_time'] = now.isoformat()
        self.session_data['duration_seconds'] = (now - self.session_start).total_seconds()
        self.session_data['total_steps'] = self.step_count
        self.session_data['total_episodes'] = self.episode_count
        self.session_data['errors'] = self.error_count
        self.session_data['warnings'] = self.warning_count

        if self.episode_rewards:
            r = self.episode_rewards
            l = self.episode_lengths
            self.session_data['episode_statistics'] = {
                'total_episodes': len(r),
                'reward': {
                    'mean': _fmean(r), 'std': _fstd(r),
                    'min': min(r), 'max': max(r),
                    'median': _fmedian(r),
                },
                'length': {
                    'mean': _fmean(l), 'std': _fstd(l),
                    'min': min(l), 'max': max(l),
                },
            }

        # Build output dict with 'command' first so it's immediately visible
        # when opening the file — no type mutation, just a fresh ordered dict
        command: str = self.session_data.get('command', '')
        output: Dict[str, Any] = {}
        if command:
            output['command'] = command
        for key, value in self.session_data.items():
            if key != 'command':
                output[key] = value

        path = self.summary_dir / "session_summary.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    # before flushing handlers, write "no errors" banner to errors.log if error_count == 0.
    # The error handler level is ERROR so we can't use self.error_logger.info() — write directly.
    def close(self):
        """Flush all files, write summary, close handlers."""
        _internal_logger.info(f"Closing TrainingLogger (agent_id={self.agent_id})")
        try:
            self.save_session_summary()
        except Exception as e:
            _internal_logger.error(f"Failed to save session summary: {e}")

        # Write "no errors" notice (one entry per env, with env id and blank line separator)
        if self.error_count == 0:
            try:
                errors_path = self.summary_dir / "errors.log"
                _env_label = f"Env {self._instance_id}" if self._instance_id is not None else "—"
                with open(errors_path, 'a', encoding='utf-8') as _ef:
                    _ef.write(
                        f"\n"
                        f"# No errors recorded during this session\n"
                        f"# Experiment : {self.experiment_name}\n"
                        f"# Agent      : {self.agent_id}\n"
                        f"# Env        : {_env_label}\n"
                        f"# Steps      : {self.step_count}\n"
                        f"# Episodes   : {self.episode_count}\n"
                    )
            except Exception as _no_err_write_error:
                _internal_logger.debug(f"Could not write no-error notice: {_no_err_write_error}")

        label = f"Agent {self.agent_id}" if self.agent_id is not None else "Single-agent"
        print(f"\n{'=' * 70}")
        print(f"LOGS SAVED [{label}]: {self.log_dir}")
        print(f"Steps: {self.step_count:,}  |  Episodes: {self.episode_count}")
        print(f"{'=' * 70}\n")


# =========================================================================
#  HELPERS INTERNES  (pas de dépendance numpy)
# =========================================================================
class SafeFormatter(logging.Formatter):
    """Formatter that injects default values for missing %(step)s / %(episode)s fields.
    Required because errors.log and debug.log handlers are attached to ALL mh_* loggers
    via register_file_handler(), but only TrainingLogger methods pass extra={'step':...}.
    Any mh_* logger.info("xxx") would crash with KeyError: 'step' without this."""
    _FIELD_DEFAULTS = {'step': '-', 'episode': '-'}

    def format(self, record):
        # Inject missing fields so %(step)s never raises KeyError
        for field, default in self._FIELD_DEFAULTS.items():
            if not hasattr(record, field):
                setattr(record, field, default)
        return super().format(record)

def _make_isolated_logger(name: str, level: int) -> logging.Logger:
    """Logger Python isolé (propagate=False, aucun handler existant)."""
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.handlers  = []
    lg.propagate = False
    return lg

def _fmean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _fstd(xs):
    if len(xs) < 2:
        return 0.0
    m = _fmean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def _fmedian(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return (s[mid] + s[mid - 1]) / 2.0 if n % 2 == 0 else float(s[mid])


# =========================================================================
#  SB3 CALLBACK
# =========================================================================

class LoggingCallback(BaseCallback):
    """
    Callback SB3 qui alimente un TrainingLogger.

    Deux modes d'utilisation :
      1. Single-agent / SB3 learn() :
         SB3 appelle _on_step() automatiquement après chaque step.

      2. Multi-agent / MultiAgentTrainer (boucle custom) :
         MultiAgentTrainer appelle record_step(reward, action, done, info)
         directement, avec les données de env.step(). Pas de dépendance
         à self.locals (uniquement valide dans SB3 learn()).
    """

    def __init__(
        self,
        training_logger: TrainingLogger,
        agent_id: Optional[int] = None,
        all_env_loggers: Optional[List['TrainingLogger']] = None,  # one per env
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.training_logger = training_logger
        # all_env_loggers[i] is the logger for env i; falls back to training_logger
        self.all_env_loggers = all_env_loggers or [training_logger]
        self.agent_id        = agent_id
        self._ep_reward      = 0.0
        self._ep_length      = 0
        self._ep_data:  Dict[str, Any] = {}
        self._ep_breakdown:  Dict[str, float] = {}  # Cumulative reward per category per episode

    # ── SB3 learn() ───────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        """Set agent context so AgentAwareFormatter prefixes 'Agent X |' on all mh_* logs."""
        if self.agent_id is not None:
            from info.agent_context import AgentContext
            AgentContext.set_current_agent(self.agent_id)

    def _on_training_end(self) -> None:
        """Clear agent context when training stops."""
        from info.agent_context import AgentContext
        AgentContext.clear()

    def _on_step(self) -> bool:
        """Called by SB3 at each step. Re-sets agent context in case thread changed."""
        if self.agent_id is not None:
            from info.agent_context import AgentContext
            AgentContext.set_current_agent(self.agent_id)

        _infos = self.locals.get('infos')
        _rewards = self.locals.get('rewards')
        _actions = self.locals.get('actions')
        _dones = self.locals.get('dones')

        # process each env with its own logger
        _infos_list = _infos if _infos is not None else [{}]
        _rewards_list = _rewards if _rewards is not None else [0.0]
        _actions_list = _actions if _actions is not None else [0]
        _dones_list = _dones if _dones is not None else [False]

        for _env_idx, (_info, _reward, _action, _done) in enumerate(
                zip(_infos_list, _rewards_list, _actions_list, _dones_list)
        ):
            _env_logger = (
                self.all_env_loggers[_env_idx]
                if _env_idx < len(self.all_env_loggers)
                else self.training_logger
            )
            # Set EnvContext so _EnvContextFilter routes logs to the correct env file.
            # Without this, all logs go to env_0 because EnvContext is None during learn().
            from info.agent_context import AgentContext, EnvContext
            if self.agent_id is not None:
                AgentContext.set_current_agent(self.agent_id)
            EnvContext.set_current_env(_env_idx)

            _orig = self.training_logger
            self.training_logger = _env_logger
            self._process_step(
                reward=float(_reward),
                action=_action.tolist() if hasattr(_action, 'tolist') else _action,
                done=bool(_done),
                info=_info,
            )
            self.training_logger = _orig

            # Clear env context after processing to avoid bleed between iterations
            EnvContext.clear()

        return True

    # ── MultiAgentTrainer direct-call ─────────────────────────────────────

    def record_step(
        self,
        reward:  float,
        action,             # <- int or list (multi-head)
        done:    bool,
        info:    Dict[str, Any],
        env_idx: int = 0,
    ):
        """
        Called from MultiAgentTrainer after each env.step() for this agent.
        env_idx selects the right per-env logger (reward_debug_envX / training_data_envX).
        Falls back to training_logger if all_env_loggers is not populated.
        """
        from info.agent_context import AgentContext, EnvContext
        if self.agent_id is not None:
            AgentContext.set_current_agent(self.agent_id)
        # Always set EnvContext so _EnvContextFilter on console_envX.log routes correctly
        EnvContext.set_current_env(env_idx)

        # Route to the correct per-env logger, same mechanism as _on_step
        _env_logger = (
            self.all_env_loggers[env_idx]
            if env_idx < len(self.all_env_loggers)
            else self.training_logger
        )
        _orig = self.training_logger
        self.training_logger = _env_logger
        self._process_step(reward=reward, action=action, done=done, info=info)
        self.training_logger = _orig

    # ── Logique commune ───────────────────────────────────────────────────

    # extract reward breakdown from info and forward to debug.log dedicated methods.
    def _process_step(self, reward, action, done, info):
        step_data: Dict[str, Any] = {
            'reward': reward,
            'action': action,
            'hp': info.get('current_hp', info.get('hp', 0)) or 0,
            'stamina': info.get('current_stamina', info.get('stamina', 0)) or 0,
            'zone': info.get('current_zone', 0) or 0,
            'hit_count': info.get('hit_count', 0) or 0,
            'death_count': info.get('death_count', 0) or 0,
        }
        if self.agent_id is not None:
            step_data['agent_id'] = self.agent_id

        self.training_logger.log_step_data(step_data)

        # --- Reward breakdown → debug.log (every step, all categories + subcategories) ---
        breakdown = info.get('reward_breakdown', {})
        breakdown_detailed = info.get('reward_breakdown_detailed', {})
        self.training_logger.log_reward_step(
            reward=reward,
            breakdown=breakdown,
            breakdown_detailed=breakdown_detailed,
        )
        # Accumulate per-episode breakdown for episode summary on reset
        for cat, val in breakdown.items():
            self._ep_breakdown[cat] = self._ep_breakdown.get(cat, 0.0) + val

        self._ep_reward += reward
        self._ep_length += 1
        self._ep_data = step_data.copy()

        # Force Python bool to avoid NumPy ambiguity
        if bool(done):
            ep_info = info.get('episode', {})
            self.training_logger.log_episode_reward_summary(
                episode=self.training_logger.episode_count + 1,
                total_reward=self._ep_reward,
                breakdown_accumulator=self._ep_breakdown,
            )
            self.training_logger.log_episode_end({
                'total_reward': float(ep_info.get('r', self._ep_reward)),
                'length': int(ep_info.get('l', self._ep_length)),
                'hits': int(info.get('hit_count', 0)),
                'deaths': int(info.get('death_count', 0)),
                'zones_discovered': int(info.get('zones_discovered', 0)),
            })
            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_data = {}
            self._ep_breakdown = {}  # Reset breakdown accumulator