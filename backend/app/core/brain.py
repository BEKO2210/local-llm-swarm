import yaml
import os
from string import Template

class PromptManager:
    def __init__(self, config_path="configs/agents.yaml"):
        # If relative, try to find it from the project root (where main.py is)
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            # Try to find it relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            potential_path = os.path.join(base_dir, "configs", "agents.yaml")
            if os.path.exists(potential_path):
                config_path = potential_path
        
        self.config_path = config_path
        self.agents = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path} (Current CWD: {os.getcwd()})")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_prompt(self, agent_name, history="", user_input="", feedback=""):
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not defined in config.")
        
        agent_cfg = self.agents[agent_name]
        template_str = agent_cfg.get("prompt_template", "")
        
        # Use simple string replacement for {{placeholders}}
        prompt = template_str.replace("{{role}}", agent_cfg.get("role", ""))
        prompt = prompt.replace("{{goal}}", agent_cfg.get("goal", ""))
        prompt = prompt.replace("{{backstory}}", agent_cfg.get("backstory", ""))
        prompt = prompt.replace("{{history}}", history)
        prompt = prompt.replace("{{input}}", user_input)
        prompt = prompt.replace("{{feedback}}", feedback)
        
        return prompt

# Singleton instance
brain = PromptManager()
