from langchain_core.language_models.llms import LLM
from vllm import LLM as VLLM
from vllm.sampling_params import SamplingParams

from core.config import config


class ChatVLLM(LLM):
    """vLLM клиент совместимый с LangChain"""

    def __init__(
        self,
        temperature=0.0,
        max_tokens=10000,
        model_name=config["model"]["name"],
        max_model_len=config["model"]["max_model_len"],
        gpu_memory_utilization=config["model"]["gpu_memory_utilization"],
        **kwargs,
    ):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm = VLLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        sampling_params = SamplingParams(
            temperature=self.temperature, max_tokens=self.max_tokens, stop=stop
        )

        return self.llm.generate([prompt], sampling_params)[0].outputs[0].text

    @property
    def _llm_type(self):
        return "vllm"
