# Batch inference for QA on the test set.
N_GPU=4
DATA_ROOT="data"
DATA_NAME="hotpotqa" # hotpotqa musique 2wikimultihopqa
LLM="gpt-4o-mini"
DOC_TOP_K=5
N_THREAD=10
RETRIEVED_RESULT_PATH="outputs/qa_finetune/latest/predictions_${DATA_NAME}_test.json"
NODE_PATH="${DATA_ROOT}/${DATA_NAME}_test/processed/stage1/nodes.csv"

# Optional: use a vLLM OpenAI-compatible endpoint for final answer generation.
LLM_BACKEND="${LLM_BACKEND:-openai}" # openai or vllm
LLM_API_BASE="${LLM_API_BASE:-}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
VLLM_ARGS=""
if [ "${LLM_BACKEND}" = "vllm" ]; then
    VLLM_ARGS="llm._target_=gfmrag.llms.VLLMChat llm.api_base=${LLM_API_BASE} llm.api_key=${LLM_API_KEY}"
fi

HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.qa \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    llm.model_name_or_path=${LLM} \
    test.n_threads=${N_THREAD} \
    test.top_k=${DOC_TOP_K} \
    test.target_types=[document] \
    test.retrieved_result_path=${RETRIEVED_RESULT_PATH} \
    test.node_path=${NODE_PATH} \
    ${VLLM_ARGS}
