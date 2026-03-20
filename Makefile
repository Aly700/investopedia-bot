SHELL := /bin/bash

.PHONY: build-quality build-growth build-broad monitor monitor-growth monitor-broad summary review test-fast

build-quality:
	./scripts/build_quality_universe.sh

build-growth:
	./scripts/build_growth_universe.sh

build-broad:
	./scripts/build_broad_universe.sh

monitor:
	./scripts/monitor_quality.sh

monitor-growth:
	./scripts/monitor_growth.sh

monitor-broad:
	./scripts/monitor_broad.sh

summary:
	./scripts/daily_quality_summary.sh

review:
	./scripts/review_portfolio.sh

test-fast:
	./scripts/test_fast.sh
