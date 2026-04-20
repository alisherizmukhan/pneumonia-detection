.PHONY: install train-all train-baseline train-resnet18 train-densenet121 \
        evaluate evaluate-all kfold gradcam collect-results run-all \
        test demo clean help

FOLDS ?= 5

help:
	@echo "Pneumonia Detection — Available commands:"
	@echo ""
	@echo "  make run-all             Full pipeline: train → evaluate → kfold → gradcam → csv"
	@echo ""
	@echo "  make install             Install Python dependencies"
	@echo "  make train-all           Train all 3 models (Baseline+ResNet-18 parallel, then DenseNet-121)"
	@echo "  make train-baseline      Train Baseline CNN only"
	@echo "  make train-resnet18      Train ResNet-18 only"
	@echo "  make train-densenet121   Train DenseNet-121 only"
	@echo "  make evaluate            Evaluate DenseNet-121 on test set"
	@echo "  make evaluate-all        Evaluate all 3 models + produce comparison.json"
	@echo "  make kfold               Stratified k-fold CV for all models  (FOLDS=5)"
	@echo "  make gradcam             Generate Grad-CAM grids for all models"
	@echo "  make collect-results     Aggregate all results into CSV files"
	@echo "  make test                Run unit tests"
	@echo "  make demo                Launch Streamlit web demo"
	@echo "  make clean               Remove checkpoints and results"

install:
	pip install -r requirements.txt

# --- Training ---

train-baseline:
	CUDA_VISIBLE_DEVICES=0 python src/train.py --config configs/config_baseline.yaml

train-resnet18:
	CUDA_VISIBLE_DEVICES=1 python src/train.py --config configs/config_resnet18.yaml

train-densenet121:
	CUDA_VISIBLE_DEVICES=0 python src/train.py --config configs/config.yaml

train-all:
	bash scripts/train.sh

# --- Evaluation ---

evaluate:
	python src/evaluate.py --config configs/config.yaml --model checkpoints/best_model_densenet121.pt

evaluate-all:
	bash scripts/evaluate.sh

# --- K-Fold CV ---

kfold:
	bash scripts/kfold.sh $(FOLDS)

# --- Grad-CAM ---

gradcam:
	bash scripts/gradcam.sh

# --- Results aggregation ---

collect-results:
	python src/collect_results.py --results-dir results

# --- Full pipeline ---

run-all:
	bash scripts/run_all.sh $(FOLDS)

# --- Misc ---

test:
	python -m pytest tests/ -v

demo:
	streamlit run app.py

clean:
	rm -rf checkpoints/*.pt \
	       results/run_* \
	       results/errors \
	       results/*.json \
	       results/*.png \
	       results/kfold \
	       results/gradcam \
	       results/*.csv
