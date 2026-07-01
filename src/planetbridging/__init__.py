"""Planet Bridging — stream AI engine weights into Loom .entity checkpoints."""

from .bedrocks import BEDROCK_IDS, BEDROCK_PLANETS, BEDROCKS, BedrockInfo, STREAM_MODEL_IDS
from .bedrock_ladder import LadderResult, run_all_bedrock_ladders, run_bedrock_ladder
from .bedrock_smoke import SmokeResult, run_all_bedrock_smokes, run_bedrock_smoke
from .compare import compare_outputs, diff_label
from . import absorb
from . import engines
from .engines import EngineStreamResult
from .stream import (
    StreamResult,
    stream_bedrock,
    stream_cnn1,
    stream_cnn2,
    stream_cnn3,
    stream_dense,
    stream_embedding,
    stream_layernorm,
    stream_lstm,
    stream_mha,
    stream_mixer,
    stream_residual,
    stream_rmsnorm,
    stream_rnn,
    stream_swiglu,
)

try:
    from .welvet_infer import (
        import_welvet,
        infer_bedrock_entity,
        infer_dense_entity,
        infer_mixer_v1_entity,
        print_compare_ladder,
        try_infer_bedrock_entity,
        WELVET_RELOAD_BEDROCKS,
        WELVET_RELOAD_SKIP,
    )
except ImportError:  # pragma: no cover
    import_welvet = None  # type: ignore
    infer_bedrock_entity = None  # type: ignore
    infer_dense_entity = None  # type: ignore
    infer_mixer_v1_entity = None  # type: ignore
    print_compare_ladder = None  # type: ignore
    try_infer_bedrock_entity = None  # type: ignore
    WELVET_RELOAD_BEDROCKS = frozenset()  # type: ignore
    WELVET_RELOAD_SKIP = frozenset()  # type: ignore

__version__ = "0.7.3"
__all__ = [
    "__version__",
    "BEDROCK_IDS",
    "BEDROCK_PLANETS",
    "BEDROCKS",
    "BedrockInfo",
    "STREAM_MODEL_IDS",
    "StreamResult",
    "EngineStreamResult",
    "SmokeResult",
    "LadderResult",
    "engines",
    "absorb",
    "run_bedrock_ladder",
    "run_all_bedrock_ladders",
    "stream_bedrock",
    "stream_dense",
    "stream_cnn1",
    "stream_cnn2",
    "stream_cnn3",
    "stream_mha",
    "stream_lstm",
    "stream_rnn",
    "stream_layernorm",
    "stream_embedding",
    "stream_rmsnorm",
    "stream_swiglu",
    "stream_residual",
    "stream_mixer",
    "run_bedrock_smoke",
    "run_all_bedrock_smokes",
    "compare_outputs",
    "diff_label",
    "import_welvet",
    "infer_dense_entity",
    "infer_bedrock_entity",
    "try_infer_bedrock_entity",
    "infer_mixer_v1_entity",
    "print_compare_ladder",
    "WELVET_RELOAD_BEDROCKS",
    "WELVET_RELOAD_SKIP",
]
