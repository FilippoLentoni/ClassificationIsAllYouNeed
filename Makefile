UV ?= $(HOME)/.local/bin/uv
MODE ?= mock
DATASET ?= all
PY := $(UV) run python -m ciayn
ARGS := --mode $(MODE) --dataset $(DATASET)

.PHONY: setup test freeze smoke template predict payload-audit payload-examples baseline-plan baseline-run \
        baseline-fetch metrics thresholding importance explain comparison escalation experiments provenance paper paper-mock verify-paper report fresh-check backup all

setup:
	@command -v $(UV) >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) python install 3.11
	$(UV) sync --frozen || $(UV) sync

test:
	$(UV) run pytest

freeze:            ## evaluation identifiers and dataset documentation (once, before anything else)
	$(PY) freeze

smoke:             ## one live Jev request
	$(PY) smoke --mode live

template:
	$(PY) template $(ARGS)

predict:
	$(PY) predict $(ARGS)

payload-audit:
	$(PY) payload-audit --mode live

payload-examples:
	$(PY) payload-examples --mode live

baseline-plan:     ## what the SageMaker baseline would create and run; starts nothing
	$(PY) baseline plan --mode live --dataset $(DATASET)

baseline-run:
	$(PY) baseline run --mode live --dataset $(DATASET)

baseline-fetch:
	$(PY) baseline fetch --mode live --dataset $(DATASET)

metrics:
	$(PY) metrics $(ARGS)

thresholding:
	$(PY) thresholding $(ARGS)

importance:
	$(PY) importance $(ARGS)

explain:
	$(PY) explain $(ARGS)

comparison:        ## F1 by training labels, cost and latency for the four models
	$(PY) comparison $(ARGS)

escalation:        ## Jev -> Qwen escalation, from saved predictions
	$(PY) escalation $(ARGS)

experiments: metrics thresholding importance explain comparison escalation

provenance:        ## save actual Jev and AWS usage (needs the response cache and the request log)
	$(PY) provenance --mode live

paper:             ## the paper from live results
	$(PY) paper --mode live

report:            ## docs/REPORT.md from the same saved outputs as the paper
	$(PY) report --mode live

fresh-check:       ## rebuild results and paper from tracked files only
	scripts/fresh_checkout_check.sh

paper-mock:        ## watermarked draft from mock results
	$(PY) paper --mode mock

verify-paper:
	$(PY) verify-paper --mode live

backup:
	$(PY) backup --mode live

all: predict experiments
