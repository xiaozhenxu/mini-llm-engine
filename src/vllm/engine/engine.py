
class LLMEngine:
    def __init__(self):
    
    def step(self):

    def add_request(self):

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True
    ) -> list[str]:

        for prompt, sampling_param in zip(prompts, sampling_params):
            self.add_request(prompt, sampling_param)
        
        while not self.is_finished():
            output, num_tokens = self.step()

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids