up:
	docker compose --profile ollama up -d --build

up-nim:
	docker compose --profile nim up -d --build

logs:
	docker compose logs -f --tail=200

down:
	docker compose down

smoke:
	bash scripts/smoke_test.sh
