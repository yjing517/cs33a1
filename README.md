# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```


### TinyStories full-data experiment

From the repository root in WSL, download each official split to a temporary
name. Re-run the same commands to resume an interrupted download. The filenames
used by preprocessing are published only after SHA-256 verification.

```bash
mkdir -p data
base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
for split in train valid; do
  name=TinyStoriesV2-GPT4-$split.txt
  test -f "data/$name" && continue
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "data/$name.partial" "$base/$name" || break
  case "$split" in
    train) checksum=6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443 ;;
    valid) checksum=6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d ;;
  esac
  printf '%s  %s\n' "$checksum" "data/$name.partial" | sha256sum -c - || break
  mv -n "data/$name.partial" "data/$name"
done
```

Encode the two splits with the local 10K BPE artifact. Preprocessing writes
memory-mappable uint16 arrays to a separate directory. Allow temporary space
for one extra copy of the split currently being encoded.

```bash
uv run python preprocess_data.py \
  --train data/TinyStoriesV2-GPT4-train.txt \
  --val data/TinyStoriesV2-GPT4-valid.txt \
  --output-dir outputs/lm_data/tinystories_full
```

On a CUDA machine, start with the experiment config. Command-line options such
as `--lr` and `--output-dir` override config values. Choose a distinct output
directory for each run.

```bash
uv run python train_lm.py --config configs/tinystories.json
uv run python generate.py \
  --checkpoint outputs/lm_runs/tinystories_baseline/best.pt \
  --prompt "Once upon a time" --temperature 0.8 --top-p 0.95 --max-new-tokens 256
```
