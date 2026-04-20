.PHONY: install train-all train-baseline train-resnet18 train-densenet121 \
       evaluate test clean demo help

help:
	@echo "Pneumonia Detection — Available commands:"
	@echo "  make install             Install Python dependencies"
	@echo "  make train-all           Train all 3 models (Baseline → ResNet-18 → DenseNet-121)"
	@echo "  make train-baseline      Train Baseline CNN only"
	@echo "  make train-resnet18      Train ResNet-18 only"
	@echo "  make train-densenet121   Train DenseNet-121 only"
	@echo "  make evaluate            Evaluate DenseNet-121 + model comparison"
	@echo "  make test                Run unit tests"
	@echo "  make demo                Launch Streamlit web demo"
	@echo "  make clean               Remove checkpoints and results"

install:
	pip install -r requirements.txt

train-baseline:
	python src/train.py --config configs/config_baseline.yaml

train-resnet18:
	python src/train.py --config configs/config_resnet18.yaml

train-densenet121:
	python src/train.py --config configs/config.yaml

train-all: train-baseline train-resnet18 train-densenet121

evaluate:
	python src/evaluate.py --config configs/config.yaml --model checkpoints/best_model_densenet121.pt

test:
	python -m pytest tests/ -v

demo:
	streamlit run app.py

clean:
	rm -rf checkpoints/*.pt results/run_* results/errors results/*.json results/*.png
