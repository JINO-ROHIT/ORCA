## ORCA - A Distributed Serving System for Transformer-Based Generative Models

this is my implementation for the [ORCA](https://www.usenix.org/conference/osdi22/presentation/yu) serving engine.

this paper solves four major problems -

1. when a heterogenous batch of requests is being processed, the engine has to wait for the longest request to complete. this means high latency, idle computation and waste of resources.

2. related to point 1, when a batch is being processed, no new request can be added. the new request has to wait even if there is compute available to process.

3. when a request is completed, it needs to be recycled to bring in a new request.

4. to batch uneven request lengths so attention can be computed properly. yes we can pad, but wastes resources.


ORCA proposes -

1. iterative/continous batching - batching to add in new requests during a batch and exit a request when its done without waiting for the batch to be done. 
2. selective attention - concat all the requests for linear passes, since the token interaction doesnot matter. for attention, split each request across the batch, perform attention and then concat them later. this way we make use of maximum computation. 


### Roadmap
- [x] Implement iterative batching
- [ ] Implement selective attention
- [ ] Support additional model architectures beyond Qwen family
- [ ] Add demo visualization


![ORCA Demo](./artifacts/demo.gif)

