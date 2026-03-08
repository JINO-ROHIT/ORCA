import torch

from typing import List
from dataclasses import dataclass, field
from queue import Queue

from safetensors.torch import load_file

from models.qwen.layers import Qwen3Model 
from models.qwen.utils import load_weights_into_qwen, Qwen3Tokenizer
from models.config import QWEN3_0_6B
from models.utils import KVCache

@dataclass
class Request:
    id: int = 0
    prompt: str = ""
    max_length: int = 100
    tokens: List[int] = field(default_factory = list)
    kv_cache: KVCache = field(default_factory = lambda: KVCache(n_layers = QWEN3_0_6B["n_layers"])) 
    is_completed: bool = False
    is_prefill: bool = True
    current_pos: int = 0

class Server:
    MAX_BATCH_SIZE = 10

    def __init__(self):
        self.pool = Queue()
        self.current_batch = []

        self._load_model()
        print("model loaded")
    
    def _print_batch_state(self):
        if hasattr(self, '_last_print_lines') and self._last_print_lines > 0:
            print(f"\033[{self._last_print_lines}A", end="")

        for request in self.current_batch:
            decoded = self.tokenizer.decode(request.tokens)
            print(f"[req {request.id}] {request.prompt}{decoded}\033[K")  # \033[K clears rest of line
        
        self._last_print_lines = len(self.current_batch)

    def _load_model(self):
        self.model = Qwen3Model(QWEN3_0_6B)
        self.model.eval()

        weights_dict = load_file("weights/Qwen3-0.6B-Base/model.safetensors")
        load_weights_into_qwen(self.model, QWEN3_0_6B, weights_dict)
        self.model.to("cuda")

        self.tokenizer = Qwen3Tokenizer(
            tokenizer_file_path = "weights/Qwen3-0.6B-Base/tokenizer.json",
            apply_chat_template = False,
            add_generation_prompt = False,
            add_thinking = False
        )
    
    
    def add_request(self, request: Request):
        """adds a request in the pool"""
        self.pool.put(request)
    
    def prefill(self, request: Request):
        """this performs the prefill stage"""
        token_ids = self.tokenizer.encode(request.prompt)
        token_ids = torch.tensor(token_ids, device = "cuda").unsqueeze(0)
        logits = self.model(token_ids, cache = request.kv_cache, current_pos = request.current_pos)
        request.current_pos += token_ids.shape[1] # advance by prompt length

        next_token = torch.argmax(logits[:, -1], dim = -1, keepdim = True)
        token = next_token.item()

        request.tokens.append(token)

        request.is_prefill = False

        if token in [151645, 151643] or len(request.tokens) >= request.max_length:
            request.is_completed = True

    def decode(self, request: Request):
        """this performs one step of decode, feed only the last token and the cache handles history"""
        logits = self.model(torch.tensor(request.tokens[-1], device = "cuda").unsqueeze(0).unsqueeze(0), cache = request.kv_cache, current_pos = request.current_pos)
        request.current_pos += 1 # single position for single token

        next_token = torch.argmax(logits[:, -1], dim = -1, keepdim = True)
        token = next_token.item()

        request.tokens.append(token)

        if token in [151645, 151643] or len(request.tokens) >= request.max_length:
            request.is_completed = True
    
    def _get_next_batch(self):
        self.current_batch = [_req for _req in self.current_batch if not _req.is_completed] # in the current batch keep the ones not completed

        #now to fill the remaining gap, add many new requests from the pool
        while not self.pool.empty() and len(self.current_batch) < Server.MAX_BATCH_SIZE:
            self.current_batch.append(self.pool.get())
        
        return self.current_batch

    def serve(self):
        self.current_batch = self._get_next_batch()
        if not self.current_batch:
            return False

        for request in self.current_batch:
            if request.is_prefill:
                self.prefill(request)
            else:
                self.decode(request)
        
        self._print_batch_state()
        return True
        
        
if __name__ == '__main__':
    server = Server()
    server.add_request(Request(id = 0, prompt = "explain what is AGI"))
    server.add_request(Request(id = 1, prompt = "write a bit about vllm"))
    server.add_request(Request(id = 2, prompt = "tell me something about sglang"))

    while server.serve():
        pass


