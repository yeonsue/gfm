# IRCoT + GFM-RAG inference on QA tasks
N_GPU=1
DATA_ROOT="data"
DATA_NAME="hotpotqa" # hotpotqa musique 2wikimultihopqa
LLM="gpt-4o-mini"
MAX_STEPS=3
MAX_SAMPLE=-1
MODEL_PATH=rmanluo/GFM-RAG-8M

# Optional: use local checkpoints and a vLLM OpenAI-compatible endpoint.
LLM_BACKEND="${LLM_BACKEND:-openai}" # openai or vllm
LLM_API_BASE="${LLM_API_BASE:-}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
EL_MODEL_PATH="${EL_MODEL_PATH:-}"
VLLM_ARGS=""
if [ "${LLM_BACKEND}" = "vllm" ]; then
    VLLM_ARGS="llm._target_=gfmrag.llms.VLLMChat llm.api_base=${LLM_API_BASE} llm.api_key=${LLM_API_KEY} ner_model.llm_api=vllm ner_model.model_name=${LLM} ner_model.api_base=${LLM_API_BASE} ner_model.api_key=${LLM_API_KEY}"
fi
if [ -n "${EL_MODEL_PATH}" ]; then
    VLLM_ARGS="${VLLM_ARGS} el_model.model_name_or_path=${EL_MODEL_PATH}"
fi

HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.qa_ircot_inference \
    dataset.root=${DATA_ROOT} \
    llm.model_name_or_path=${LLM} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${DATA_NAME} \
    agent_prompt=${DATA_NAME}_ircot \
    test.max_steps=${MAX_STEPS} \
    test.max_test_samples=${MAX_SAMPLE} \
    dataset.data_name=${DATA_NAME}_test \
    graph_retriever.model_path=${MODEL_PATH} \
    ${VLLM_ARGS}
