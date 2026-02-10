from typing import Any, Mapping

import requests
from requests.exceptions import RequestException

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import threading 
import psutil 
import argparse
import torch
import torch.nn.functional as F
import gc 
import os
import re
import random
import uuid
import json

from requests.exceptions import RequestException
import requests

import torch.nn.functional as F
import torch

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool

class SciworldEnvClient:
    def __init__(self, env_server_base: str, *args, timeout: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        # self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        self.env_id = ok["id"]

    # def __len__(self):
    #    return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        response = self._post("step", {"action": action})
        try:
            self.info = {
                "observation": response["observation"],
                "reward": response["reward"],
                "score": response["score"],
                "done": response["done"],
            }
            
            return StepOutput(
            state=response["observation"],
            reward=response["score"],
            done=response["done"],
        )
        except:
            #print(response)

            expected_keys = ["observation", "available_actions", "reward", "done"]

            self.info = {
                "observation": "the task is finished.",
                "available_actions": [],
                "reward": 1.0,
                "done": True,
            }
            for key in expected_keys:
                if key in response:
                    self.info[key] = response[key]
                else:
                    self.info[key] = "missing"
                    #response[key] = "missing"
                    #print(f"Warning: key '{key}' missing in response")
            
            return StepOutput(
                state=self.info["observation"],
                reward=1.0,
                done=self.info["done"],
            )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["task_description"]
            + "\n"
            + response["observation"],
            "reward": 0,
            "score": 0,
            "done": False,
        }
        return response

class AlfWorldEnvClient:
    def __init__(
        self,
        env_server_base: str,
        *args,
        # data_len: int = 4096, # retired
        timeout: int = 300,
        **kwargs,
    ):
        self.env_server_base = env_server_base
        self.timeout = timeout

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        self.env_id = ok["id"]
        self.info = None

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return f"{self.info['observation']}\nAVAILABLE ACTIONS: {','.join(self.info['available_actions'])}"

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        #print(f"Action: {action}")
        response = self._post("step", {"action": action})
        #print(response)

        try:
            self.info = {
                "observation": response["observation"],
                "available_actions": response["available_actions"],
                "reward": response["reward"],
                "done": response["done"],
            }
            return StepOutput(
                state=response["observation"],
                reward=response["reward"],
                done=response["done"],
            )
        except:
            #print(response)

            expected_keys = ["observation", "available_actions", "reward", "done"]

            self.info = {
                "observation": "the task is finished.",
                "available_actions": [],
                "reward": 1.0,
                "done": False,
            }
            for key in expected_keys:
                if key in response:
                    self.info[key] = response[key]
                else:
                    self.info[key] = "missing"
                    #response[key] = "missing"
                    #print(f"Warning: key '{key}' missing in response")
            
            
            
            return StepOutput(
                state=self.info["observation"],
                reward=1.0,
                done=self.info["done"],
            )

    def reset(self, game: int, world_type: str = "Text") -> dict[str, Any]:
        response = self._post("reset", {"game": game, "world_type": world_type})
        self.info = {
            "observation": response["observation"],
            "available_actions": response["available_actions"],
            "reward": 0,
            "done": False,
        }
        return response

class WeatherEnvClient():

    def __init__(
        self, env_server_base: str, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        #self.data_len = data_len
        self.id = 0
        data = dict()
        data["id"] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    #def __len__(self):
    #    return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response

class MovieEnvClient():

    def __init__(
        self, env_server_base: str, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        #self.data_len = data_len
        self.id = 0
        data = dict()
        data["id"] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    #def __len__(self):
    #    return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response

class MazeEnvClient():
    _fully_first_observation = """\
Your objective is to reach the goal in as few steps as possible. At each step you will be given information about where the goal is, your current position,
and the walls that surround you. 

When you move right you increase your y position by 1, when you move down you increase your x position by 1. 

Here is an example.

```
environment: The goal is at position 8, 6. Your current position is at position 5, 6. There is a wall above you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 5. There are walls above you, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move up
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: The goal is at position 8, 6. Your current position is at position 5, 3. There are walls to your left, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 5, 4. There are walls above you, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 5. There are walls above you, below you.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 5, 6. There is a wall above you.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 6, 6. There are walls to your right, to your left.
action: move down
environment: The goal is at position 8, 6. Your current position is at position 7, 6. There are walls to your right, to your left.
action: move right
environment: The goal is at position 8, 6. Your current position is at position 7, 6. There are walls to your right, to your left.
action: move down
environment: Success
```

Your possible actions are "move up", "move down", "move left", "move right". Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<Your Action>

Now let's start a new game. Return your action and your thought in the format above strictly. Now, make the optimal action given the current environment state:
""".strip()
    _partially_first_observation = """\
Your objective is to reach the goal in as few steps as possible. At each step you will see your move history, and the walls that surround you.

Here is an example. 
```
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move left
environment: There is a wall above you.
action: move left
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move up
environment: There are walls above you, below you.
action: move right
environment: There is a wall above you.
action: move down
environment: There are walls to your right, to your left.
action: move down
environment: There are walls to your right, to your left.
action: move right
environment: There are walls to your right, to your left.
action: move down
environment: Success
```

Your possible actions are "move up", "move down", "move left", "move right". Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<Your Action>

Now let's start a new game. Return your action and your thought in the format above strictly. Now, make the optimal action given the current environment state:
""".strip()
    def __init__(
        self,
        env_server_base: str,
        #data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        #self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        self.env_id = ok["id"]
        self.info = {
            "reward": 0,
            "done": False,
        }
        
    #def __len__(self):
    #    return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        print(action)
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        print(f"Action: {action}")
        response = self._post("step", {"action": action})
        print(response)
        self.info.update(
            {
                "observation": response["observation"],
                "reward": self.info["reward"] + response["reward"],
                "done": response["done"],
            }
        )
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"game": idx})
        print(response)
        self.first_observation = self._fully_first_observation
        response["observation"] = (
            self.first_observation + "\n" + response["observation"]
        )
        self.info.update(
            {
                "observation": response["observation"],
                "reward": 0,
                "done": False,
            }
        )
        return response

class WordleEnvClient():
    first_observation = """\
Welcome to the game of Wordle. Your objective is to guess a hidden 5 letter word. You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible. When guessing the word, you should format your word as a space separated sequence of letters, like "s h i r e" for example. After guessing the word, you will receive feedback from the game environment in the form of a sequence of 5 space separated letters like "b y g g b", where each letter indicates some information about the hidden word. The environment will return one of three letters – "b", "g", or "y" – for each letter in the word you guessed. We describe the meaning of each letter below:

"b": If the environment returns a "b", it means that the letter at that position in your guessed word is not in the hidden word.
"y": If the environment returns a "y", it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
"g": If the environment returns a "g", it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.

As a note, if you guess an invalid word (e.g. not a 5 letter word or a word not in the vocabulary), the environment will respond with an "invalid word" message. In general though, you should use this information returned by the environment to update your belief about what the hidden word might be and adjust your next guess accordingly.

Here is the complete list of valid vocabulary words that are accepted by the game:
```
{{vocab}}
```

Here is an example. If the current status of the game is given as:
```
guess 1: p a n i c
feedback 1: b b y b b
guess 2: f e l o n
feedback 2: g b b y g
```
Based on the feedback from the environment, you know that the first letter is "f", the last letter is "n", and there is an "o" somewhere in the word, but it is not in the second to last position. You also know that there is not a "p", "a", "i", "c", "e", or "l" in the word. Knowing this, you might guess the next word to be:
Thought:\nI know that the first letter is "f", the last letter is "n", and there is an "o" somewhere in the word, but it is not in the second to last position. I also know that there is not a "p", "a", "i", "c", "e", or "l" in the word. A good word from the vocabulary to try might therefore be \"f r o w n\", since it is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of "o". Therefore this is a good guess to try next.\n\nAction:\nf r o w n

Formally, your return should be in this format:
Thought:\n<Your Thought>\n\nAction:\n<The Word You Guess>

The guessed word is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of "o", so it is a good guess to try next.

Now let's start a new game. Remember, the word you guess should be strictly in the vocabulary. You should return your thought and your word strictly in the formation mentioned above.
""".strip()
    def __init__(
        self,
        env_server_base: str,
        #data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        #self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        #print(ok)
        self.env_id = ok["id"]
        vocab = self._get("filtered_vocab")
        self.info = {
            "observation": self.first_observation.replace(
                "{{vocab}}", "\n".join(vocab)
            ),
            "vocab": vocab,
            "reward": 0,
            "done": False,
        }
        print(self.info["observation"])

    #def __len__(self):
    #    return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        print(action)
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        print(f"Action: {action}")
        response = self._post("step", {"action": action})
        print(response)
        self.info.update(
            {
                "observation": response["observation"],
                "reward": self.info["reward"] + response["reward"],
                "done": response["done"],
            }
        )
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"seed": idx})
        self.info.update(
            {
                "observation": self.first_observation.replace(
                    "{{vocab}}", "\n".join(self.info["vocab"])
                ),
                "reward": 0,
                "done": False,
            }
        )
        return response

class TextCraftEnvClient:
    def __init__(
        self,
        env_server_base: str,
        # data_len: int = 4096, # retired
        *args,
        timeout: int = 300,
        minecraft_dir: str = "agentenv_textcraft/",
        commands: str = None,
        goal: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        # self.data_len = data_len

        dir_info = {"minecraft_dir": minecraft_dir, "commands": commands, "goal": goal}
        ok = requests.post(
            f"{self.env_server_base}/create", timeout=self.timeout, json=dir_info
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        self.env_id = ok["id"]
        self.info = {
            "observation": ok["observation"],
            "reward": 0,
            "done": False,
        }

    # def __len__(self):
    #    return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        action = action.split("Instruction:")[0].split("Action:")[-1]
        action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
        action = " ".join(action.split()).strip()
        response = self._post("step", {"action": action})
        print(response)
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": idx})
        self.info.update(
            {
                "observation": response["observation"],
                "reward": 0,
                "done": False,
            }
        )
        return response

class WeatherEnvClient():

    def __init__(
        self, env_server_base: str, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        #self.data_len = data_len
        self.id = 0
        data = dict()
        data["id"] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    #def __len__(self):
    #    return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response


class BabyAIEnvClient:
    def __init__(self, env_server_base: str, *args, timeout: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        # self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        self.env_id = ok["id"]

    # def __len__(self):
    #    return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        response = self._post("step", {"action": action})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "score": response["score"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["score"],
            done=response["done"],
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "score": response["score"],
            "done": response["done"],
        }
        return response

class Env_Conn_Full:
    def __init__(
        self,
        client,
        idx,
        save_path="traj",
        max_iter=30,
        game_type="alfworld",
        print_order = "goal|plan|observation|action|memory|attempted"
    ):
        # save parameters
        self.client = client
        self.idx = idx
        self.save_path = save_path
        self.max_iter = max_iter
        self.game_type = game_type
        self.done = False

        # init
        msg = self.client.reset(idx)
        if self.game_type == "alfworld":
            self.task_type = msg["task_type"]
            self.goal = msg["observation"].split("\n")[-1].split(":")[-1].strip()
            self.observation = msg["observation"].split("\n")[0].strip().split("Looking quickly around you")[0].strip()
        elif self.game_type == "sciworld":
            self.task_type = msg["task_name"] + "_" + str(msg["var_num"]) + "_" + str(idx)
            self.goal = msg["task_description"]
            self.observation = "None" #msg["observation"]
        elif self.game_type == "babyai":
            self.task_type = "babyai_" + str(idx)
            self.goal = msg["observation"].split("\n")[0].split(":")[-1].strip()
            self.observation = msg["observation"].split("\n")[1]
        elif self.game_type == "wordle":
            self.task_type = "wordle_" + str(idx)
            obs = client.observe()
            self.goal = "\n".join(obs.split("\n")[:7]).strip()
            self.observation = "None"
        elif self.game_type == "maze":
            self.task_type = "maze_" + str(idx)
            self.observation = msg["observation"].split("\n")[-1]
            self.goal = "\n".join(msg["observation"].split("\n")[:4])
        elif self.game_type == "movie":
            self.task_type = "movie_" + str(idx)
            self.observation = "None"
            self.goal = msg.split("\n")[1].split(":")[-1].strip()
        elif self.game_type == "weather":
            self.task_type = "weather_" + str(idx)
            self.observation = "None"
            self.goal = msg.split("\n")[1].split(":")[-1].strip()
        elif self.game_type == "textcraft":
            self.task_type = "textcraft" + str(idx)
            self.goal = msg["observation"].split("\n")[-1].replace("Goal:", "").strip()
            self.observation = "None"
        
        # subgoal
        self.plan = None 
        # attempts
        self.plan_history = None

        # observe
        msg = self.client.observe()
        if self.game_type in ["alfworld"]:
            self.action = msg.split("AVAILABLE ACTIONS:")[-1].strip()
        elif self.game_type in ["babyai"]:
            self.action = (
                msg.split("Available actions: [")[-1][:-1]
                .replace('"', "")
                .strip()
                .strip()
            )
        elif self.game_type in ["sciworld"]:
            self.action = self.client.step("look around").state
        elif self.game_type in ["wordle"]:
            self.action = obs.split("```")[1].strip()
        elif self.game_type in ["textcraft"]: #msg.find("Crafting commands:") > -1:
            self.action = msg.split("Crafting commands:")[-1].split("Goal")[0].strip()
        elif self.game_type in ["maze"]:
            self.action = "move up, move down, move left, move right"
        elif self.game_type in ["movie"]:
            self.action = "finish, get_movie_alternative_titles, get_movie_cast, get_movie_crew, get_movie_details, get_movie_keywords, get_movie_production_companies, get_movie_production_countries, get_movie_translation, get_person_cast, get_person_crew, get_person_details, get_person_external_ids, get_search_movie, get_search_person"
        elif self.game_type in ["weather"]:
            self.action = "finish, get_air_quality_level, get_current_air_quality_index, get_current_rain, get_current_snow, get_current_temp, get_distance, get_elevation, get_historical_air_quality_index, get_historical_rain, get_historical_snow, get_historical_temp, get_latitude_longitude, get_rain_forecast, get_snow_forecast, get_temp_forecast, get_user_current_date, get_user_current_location"
        else:
            self.action = "skipped"
            
        self.attempted = None
        # placeholder for last action
        self.last_step = None

        # memory
        self.memory = "None"
        if self.game_type in ["alfworld", "sciworld"]:
            msg = self.client.step("inventory").state
            self.memory = msg
        if self.game_type in ["babyai"]:
            self.memory = "You are not carrying anything."
        elif self.game_type in ["wordle"]:
            self.wordle_mem = {
                "pc": ["_"] * 5,
                "cl": [],
                "el": []
            }
            pc = " ".join(self.wordle_mem["pc"])
            cl = " ".join(self.wordle_mem["cl"])
            el = " ".join(self.wordle_mem["el"])
            self.memory = f"Known position constraints: {pc}\nConfirmed letters: {cl}\nExcluded letters: {el}"

        # iter containers
        self.block_list = print_order.split("|") #["goal", "plan", "observation", "attempted"]
        
        # create the starter
        self.history = []
        self.history.append(
            {
                "role": "user",
                "content": self.fill_template(),
                "tag": 0,  # 0 == start or instrctions
            }
        )

    def step(self, action):
        # append history
        #done = False
        action_exec = action

        # find pattern
        rule1 = action_exec.find("</proposal>") > -1
        rule2 = action_exec.find("<step>") > -1
        rule3 = action_exec.find("</step>") > -1

        # if follow the rule
        if rule1 and rule2 and rule3:
            # trigger the selected action 
            self.history.append(
                {"role": "assistant", "content": action_exec, "tag": -1} # -1 as a placeholder
            )
            # extract plan
            plan = action_exec.split("</proposal>")[0].strip()
            if plan == self.plan:
                reward_tag = 1  # 1 == same state, last step is not useful
            else:
                reward_tag = 2  # 2 == state transition successful, last step is useful
            # extract action
            self.last_step = action_exec.split("<step>")[1].split("</step>")[0].strip()
            action_send = "Action: {}".format(self.last_step)
            # take the action
            msg_action = self.client.step(action_send)            
            self.done = msg_action.done
            # observe feedback and update states
            msg = self.client.observe()
            if reward_tag == 1:
                if self.attempted is None:
                    self.attempted = self.last_step
                else:
                    self.attempted += ",{}".format(self.last_step)
            else:
                if self.plan_history is None:
                    self.plan_history = plan
                else:
                    self.plan_history += ",{}".format(plan)
                self.plan = plan
                self.attempted = self.last_step
            
            if not self.done:
                if self.game_type == "alfworld":
                    self.observation = msg.split("AVAILABLE ACTIONS:")[0].strip()
                    msg_mem = self.client.step("inventory").state
                    self.memory = msg_mem
                    self.action = msg.split("AVAILABLE ACTIONS:")[-1].strip()
                elif self.game_type == "textcraft":
                    self.observation = msg#["observation"]
                    msg_mem = self.client.step("inventory").state
                    self.memory = msg_mem.replace("Inventory:", "").strip()
                elif self.game_type == "babyai":
                    msg = self.client.observe().split("\n")[0]
                    msg = msg.split("<observation>")[-1].split("</observation>")[0].split(".")
                    if len(msg) > 2:
                        temp_mem = msg[-2].strip() + "."
                        if temp_mem != "Please check valid actions.":
                            self.memory = temp_mem
                elif self.game_type == "sciworld":
                    self.observation = msg
                    msg_mem = self.client.step("inventory").state
                    self.memory = msg_mem
                    msg_lr = self.client.step("look around").state
                    self.action = msg_lr
                elif self.game_type == "maze":
                    self.observation = msg_action.state
                elif self.game_type in ["movie", "weather"]:
                    self.observation = msg_action.state.replace("Give me one action.", "").strip()
                    if self.memory == "None":
                        self.memory = ""
                    self.memory += f"Action: {self.last_step}\n{self.observation}\n\n"
                elif self.game_type == "wordle":
                    #'b y b b b\n'
                    #self.wordle_mem = {
                    #    "pc": ["_"] * 5,
                    #    "cl": [],
                    #    "el": []
                    #}
                    if not msg == "invalid word":
                        msg_processed = msg.strip().split(" ")
                        action_processed = self.last_step.strip().split(" ")
                        for lpos, letter in enumerate(msg_processed):
                            if letter == "g":
                                self.wordle_mem["pc"][lpos] = action_processed[lpos]
                                action_processed[lpos] not in self.wordle_mem["cl"] and self.wordle_mem["cl"].append(action_processed[lpos])
                            elif letter == "y":
                                action_processed[lpos] not in self.wordle_mem["cl"] and self.wordle_mem["cl"].append(action_processed[lpos])
                            elif letter == "b":
                                action_processed[lpos] not in self.wordle_mem["el"] and self.wordle_mem["el"].append(action_processed[lpos])
                    pc = " ".join(self.wordle_mem["pc"])
                    cl = " ".join(self.wordle_mem["cl"])
                    el = " ".join(self.wordle_mem["el"])
                    self.memory = f"Known position constraints: {pc}\nConfirmed letters: {cl}\nExcluded letters: {el}"
                elif msg.find("Available actions: [") > -1:
                    self.observation = msg.split("Available actions: [")[0].strip()
                    self.action = (
                        msg.split("Available actions: [")[-1][:-1].replace('"', "").strip()
                    )
            else:#     if self.done:
                reward_tag = 3  # 3 == task successful, last step is useful
                
            # update history
            self.history.append(
                {
                    "role": "user",
                    "content": self.fill_template(),
                    "tag": reward_tag,  # either 1 or 2
                }
            )
            if self.done:
                self.save(True)
        # if not follow the rule
        else:
            # update history with failed attempt
            self.history.append(
                {"role": "user", "content": action, "tag": -2}  # not allowed actions
            )
            self.observation = "Nothing happens."
            self.history.append(
                {
                    "role": "assistant",
                    "content": self.fill_template(),
                    "tag": -2,  # not allowed actions
                }
            )

        # if max_iter reached
        if (not self.done) and (len(self.history) > 2 * self.max_iter):
            self.save(False)
            self.done = True  # exhausted
        return self.done  # Ture if finished or exhausted

    def fill_template(self):
        # fill template
        template = ""
        #self.content_list = [getattr(self, block) for block in self.block_list]
        self.content_list = [getattr(self, block) if block != "plan" else getattr(self, "plan_history") for block in self.block_list]
        for block, content in zip(self.block_list, self.content_list):
            template += "<{}>\n{}\n</{}>\n\n".format(block, str(content), block)
        template += "<proposal>\n"
        self.template = template
        return template

    def save(self, status):
        json_obj = {
            "task_type": self.task_type,
            "id": self.idx,
            "conversations": self.history,
        }
        if type(self.task_type) == str:
            task_type = self.task_type
        else:
            task_type = str(self.idx)
        with open(
            "{}/{}-{}-{}.json".format(
                self.save_path,
                task_type.replace("/", "_"),
                str(status),
                "man"#str(uuid.uuid4()).split("-")[0],
            ),
            "w",
        ) as handle:
            json.dump(json_obj, handle)


@dataclass
class StepOutput:
    state: str
    reward: float
    done: bool

class AlfWorldEnvClient():
    def __init__(
        self,
        env_server_base: str,
        *args,
        #data_len: int = 4096, # retired
        timeout: int = 300,
        **kwargs,
    ):
        self.env_server_base = env_server_base
        self.timeout = timeout

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        print(ok)
        self.env_id = ok["id"]
        self.info = None

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return f"{self.info['observation']}\nAVAILABLE ACTIONS: {','.join(self.info['available_actions'])}"

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        _action = action.split("Action:")
        if len(_action) > 1:
            action = _action[1].strip()
        else:
            action = _action[0].strip()
        print(f"Action: {action}")
        response = self._post("step", {"action": action})
        print(response)
        self.info = {
            "observation": response["observation"],
            "available_actions": response["available_actions"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, game: int, world_type: str = "Text") -> dict[str, Any]:
        response = self._post("reset", {"game": game, "world_type": world_type})
        self.info = {
            "observation": response["observation"],
            "available_actions": response["available_actions"],
            "reward": 0,
            "done": False,
        }
        return response