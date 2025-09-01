python score_split.py \
  --csv_path "dataset/mini_train.csv" \
  --output_dir "split_lines" \
  --max_pages 50 

python score_split.py \
  --csv_path "dataset/mini_test.csv" \
  --output_dir "split_lines" \
  --max_pages 50 

# python score_split.py \
#   --csv_path "filtered_music_scores.csv" \
#   --output_dir "split_lines" \
#   --max_pages 10 \
#   --max_samples 10