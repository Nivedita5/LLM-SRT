CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  inference_seamless_m4t.py \
  --model_dir seamless_test_run \
  --test_manifest /path/to/test/manifest.json \
  --source_lang eng \
  --target_lang hin \
  --batch_size 4 \
  --output_file results.tsv