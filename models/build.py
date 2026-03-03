# pretty shit design, refactor later to scale
# not used as of now

from qwen.layers import Qwen3Model

from config import QWEN3_0_6B


class OrcaModel:
    def __call__(self, model_name: str):
        if 'qwen' in model_name:
            model = Qwen3Model(QWEN3_0_6B)
        return model


    