s-sushi:
	python scripts/cron/schedule_sushi.py

s-metrics:
	python scripts/cron/schedule_metrics.py

i-metrics:
	python scripts/influx_metrics.py

test:
	pytest

test-watch:
	watchexec -c pytest

shell:
	poetry shell
