# NOTE: ChatGPT Prompt
# Write a bash script that does a grid search on the top-k top-p and temperature parameters

SENTENCE="Two boys are sitting on a couch that is red and black"
CHECKPOINT="nmt.model"
CONFIG="../part2/model_config.yaml"
VOCAB="../part2/vocab/vocab.json"
ENCODER="../part2/vocab/spm.model"

# Define search ranges
TOP_K_VALUES=(2 4 8 16)
TOP_P_VALUES=(0.2 0.4 0.6 0.8)
TEMPERATURE_VALUES=(0.25 0.5 0.75 1.0)

OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"

for k in "${TOP_K_VALUES[@]}"; do
  for p in "${TOP_P_VALUES[@]}"; do
    for t in "${TEMPERATURE_VALUES[@]}"; do

      # Build a safe filename
      FILE="$OUTPUT_DIR/k${k}_p${p}_t${t}.txt"

      echo "Running k=$k  p=$p  t=$t → $FILE"

      python ../part2/predict.py \
        --sentence "$SENTENCE" \
        --checkpoint-path $CHECKPOINT \
        --model-config $CONFIG \
        --vocab-file $VOCAB \
	--encoder $ENCODER \
        --do-sample \
        --top-k $k \
        --top-p $p \
        --temperature $t \
        > "$FILE"

    done
  done
done

echo "Results saved under $OUTPUT_DIR/"
