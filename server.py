'''
basic scaffold to implement
1. have a pool of requests
2. collect a batch(assume my gpu can do 10 requests max)
3. if request.is_completed, then remove from the batch, and fetch a new from pool
'''
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
    max_length: int = 2048
    tokens: List[int] = field(default_factory = list)
    kv_cache: KVCache = KVCache(n_layers = QWEN3_0_6B["n_layers"])
    is_completed: bool = False
    is_prefill: bool = True # do we need this or nah?

class Server:
    MAX_BATCH_SIZE = 10

    def __init__(self):
        self.pool = Queue()
        self.current_batch = []

        self._load_model()
        print("model loaded")
    
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
    
    def _tokenizer_decode(self, request: Request):
        for token in request.tokens:
            print(f"{request.id} - {self.tokenizer.decode([token])}")
    
    def add_request(self, request: Request):
        """adds a request in the pool"""
        self.pool.put(request)
    
    def prefill(self, request: Request):
        """this performs the prefill stage"""
        token_ids = self.tokenizer.encode(request.prompt)
        token_ids = torch.tensor(token_ids, device = "cuda").unsqueeze(0)
        logits = self.model(token_ids, cache = request.kv_cache)

        next_token = torch.argmax(logits[:, -1], dim = -1, keepdim = True)
        token = next_token.item()

        request.tokens.append(token)

        request.is_prefill = False

        self._tokenizer_decode(request)

        if token in [151645, 151643] or len(request.tokens) >= request.max_length:
            request.is_completed = True

    def decode(self, request: Request):
        """this performs one step of decode, feed only the last token and the cache handles history"""
        logits = self.model(request.tokens[-1], cache = request.kv_cache)

        next_token = torch.argmax(logits[:, -1], dim = -1, keepdim = True)
        token = next_token.item()

        request.tokens.append(token)

        self._tokenizer_decode(request)

        if token in [151645, 151643] or len(request.tokens) >= request.max_length:
            request.is_completed = True


    def serve(self):

        while self.pool.empty() is False and len(self.current_batch) > 1:
            request = self.pool.get()

            if len(self.current_batch) < Server.MAX_BATCH_SIZE:
                self.current_batch.append(request)
            

            for request in self.current_batch:
                if request.is_completed:
                    self.current_batch.remove(request)

                if request.is_prefill:
                    self.prefill(request)
                else:
                    self.decode(request)
        
        print("done serving all requests")
        
        
if __name__ == '__main__':
    server = Server()
    server.add_request(Request(prompt = "california is"))
    server.serve()


