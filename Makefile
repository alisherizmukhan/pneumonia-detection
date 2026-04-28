.PHONY: train evaluate interpret analyze all \
        setup-eval install test demo clean help

# ── Primary targets (as specified) ───────────────────────────────────────────

train:
	bash scripts/train_all.sh

evaluate:
	python src/evaluate.py

interpret:
	python src/run_interpretability.py

analyze:
	python src/analysis.py

all: train evaluate interpret analyze

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

setup-eval:
	bash scripts/setup_eval_subset.sh

# ── Testing & demo ────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

demo:
	streamlit run app.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf checkpoints/*.pt \
	       results/run_* \
	       results/errors \
	       results/*.json \
	       results/*.png \
	       results/*.csv \
	       results/kfold \
	       results/gradcam \
	       results/interpretability

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Pneumonia Detection — Multi-Model XAI Pipeline"
	@echo ""
	@echo "  make all          Full pipeline: train → evaluate → interpret → analyze"
	@echo "  make train        Train all 4 models (DenseNet121, ResNet18, EfficientNet-B0, MobileNetV2)"
	@echo "  make evaluate     Evaluate all models, save per-model metrics + final_comparison.csv"
	@echo "  make interpret    Run Grad-CAM, LRP, and Occlusion on eval subset"
	@echo "  make analyze      Assign lung-focus verdicts and update final_comparison.csv"
	@echo ""
	@echo "  make setup-eval   Copy 5+5 images from test set to data/eval_subset/"
	@echo "  make install      Install Python dependencies"
	@echo "  make test         Run unit tests"
	@echo "  make demo         Launch Streamlit web demo"
	@echo "  make clean        Remove checkpoints and results"
	@echo ""
