# Build the index for testing dataset
N_GPU=1
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_test 2wikimultihopqa_test musique_test"

# Optional: use a vLLM OpenAI-compatible endpoint for NER/OpenIE.
LLM_API="${LLM_API:-openai}" # openai or vllm
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
LLM_API_BASE="${LLM_API_BASE:-}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
EL_MODEL_PATH="${EL_MODEL_PATH:-}"
VLLM_ARGS=""
if [ "${LLM_API}" = "vllm" ]; then
   VLLM_ARGS="ner_model.llm_api=vllm ner_model.model_name=${LLM_MODEL} ner_model.api_base=${LLM_API_BASE} ner_model.api_key=${LLM_API_KEY} openie_model.llm_api=vllm openie_model.model_name=${LLM_MODEL} openie_model.api_base=${LLM_API_BASE} openie_model.api_key=${LLM_API_KEY}"
fi
if [ -n "${EL_MODEL_PATH}" ]; then
   VLLM_ARGS="${VLLM_ARGS} el_model.model_name_or_path=${EL_MODEL_PATH}"
fi

for DATA_NAME in ${DATA_NAME_LIST}; do
   HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.index_dataset \
   dataset.root=${DATA_ROOT} \
   dataset.data_name=${DATA_NAME} \
   ${VLLM_ARGS}
done


# Build the index for training dataset

N_GPU=1
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train" #
START_N=0
END_N=19
for i in $(seq ${START_N} ${END_N}); do
   for DATA_NAME in ${DATA_NAME_LIST}; do
      HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.index_dataset \
      dataset.root=${DATA_ROOT} \
      dataset.data_name=${DATA_NAME}${i} \
      ${VLLM_ARGS}
   done
done
