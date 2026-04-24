# Language Learning Assistant - Makefile
# Run `make <target>` to execute a target

SHELL := /bin/zsh
PYTHON := .venv/bin/python3
UV := uv
PNPM := pnpm

# Check that venv exists
ifeq ($(wildcard .venv/bin/activate),)
  $(error Virtual environment not found. Run: make setup)
endif

.PHONY: help install run-web run-server dev clean test web-build web-lint \
	docker-up docker-down docker-logs docker-rebuild docker-agent-dev docker-ps all

help:                                    ## Show this help message
	@echo "Language Learning Assistant"
	@echo "============================"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make setup        (first time only)"
	@echo "  2. make dev          (run both web + backend)"
	@echo ""

setup:                                   ## First-time setup (creates venv, installs deps)
	@echo "Setting up Python environment..."
	@$(UV) venv --python 3.12
	@echo "Installing Python dependencies..."
	@$(UV) pip install -e ".[dev]"
	@echo ""
	@echo "Creating .env file..."
	@test -f .env || cp .env.example .env
	@echo "⚠️  Add your OPENROUTER_API_KEY to .env"
	@echo ""
	@echo "Installing web dependencies..."
	@cd web && $(PNPM) install
	@echo ""
	@echo "Setup complete!"

install:                                 ## Install all dependencies
	@echo "Installing Python dependencies..."
	@$(UV) pip install -e ".[dev]"
	@echo "Installing web dependencies..."
	@cd web && $(PNPM) install

run-server:                              ## Run Python bridge server (localhost:8000)
	@echo "Starting bridge server..."
	@$(PYTHON) src/server.py

run-web:                                 ## Run Next.js web app (localhost:3000)
	@echo "Starting web app..."
	@cd web && $(PNPM) dev

dev:                                     ## Run both web + server (background server, foreground web)
	@echo "Starting both web + server..."
	@echo "Starting bridge server in background..."
	@nohup $(PYTHON) src/server.py > server.log 2>&1 & echo $$! > .server.pid
	@echo "Server started (PID: $$(cat .server.pid))"
	@echo ""
	@echo "Starting web app..."
	@cd web && $(PNPM) dev

stop:                                    ## Stop background server
	@test -f .server.pid && kill $$(cat .server.pid) 2>/dev/null && rm -f .server.pid || echo "No server running"

test:                                    ## Run Python tests
	@echo "Running tests..."
	@$(PYTHON) -m pytest tests/ -v

web-build:                               ## Build Next.js app
	@echo "Building web app..."
	@cd web && $(PNPM) build

web-lint:                                ## Lint TypeScript code
	@echo "Linting web app..."
	@cd web && $(PNPM) lint

clean:                                   ## Clean build artifacts
	@echo "Cleaning..."
	@rm -rf web/.next web/node_modules web/tsconfig.tsbuildinfo
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov
	@rm -f .server.pid server.log
	@echo "Cleaned!"

# ─── Development helpers ───────────────────────────────────────────────────────

check:                                   ## Check env and deps are ready
	@echo "Checking environment..."
	@$(PYTHON) -c "import fastapi, uvicorn, openai, dotenv; print('✓ Python deps OK')" 2>/dev/null || echo "✗ Run: make setup"
	@test -f .env || echo "✗ Run: cp .env.example .env && add your OPENROUTER_API_KEY"
	@grep -q "sk-or-v1-" .env 2>/dev/null && echo "✓ API key set" || echo "⚠️  OPENROUTER_API_KEY not found in .env"
	@cd web && $(PNPM) -v >/dev/null 2>&1 && echo "✓ Web deps OK" || echo "✗ Run: cd web && pnpm install"

health:                                  ## Check if backend server is running
	@curl -s http://localhost:8000/api/health 2>/dev/null | python3 -m json.tool || echo "Backend not running"

# ─── Docker / LiveKit (Phase 2) ────────────────────────────────────────────────

docker-up:                               ## Start LiveKit server + voice agent
	@echo "Starting LiveKit server + voice agent..."
	@docker compose up -d --build
	@echo ""
	@echo "Services running:"
	@echo "  LiveKit server:  http://localhost:7880"
	@echo "  Voice agent:     (connects to LiveKit, health: localhost:8081)"
	@echo ""
	@echo "Test with:"
	@echo "  make docker-logs     # view logs"
	@echo "  make docker-ps       # check status"
	@echo ""
	@echo "Connect from web app by setting:"
	@echo "  LIVEKIT_URL=ws://localhost:7880"
	@echo "  LIVEKIT_API_KEY=devkey"
	@echo "  LIVEKIT_API_SECRET=secret"

docker-down:                             ## Stop LiveKit + voice agent
	@echo "Stopping LiveKit + voice agent..."
	@docker compose down

docker-logs:                             ## View LiveKit + voice agent logs
	@docker compose logs -f

docker-rebuild:                          ## Rebuild and restart voice agent
	@echo "Rebuilding voice agent..."
	@docker compose up -d --build voice-agent

docker-agent-dev:                        ## Run voice agent in dev mode (hot reload)
	@echo "Starting voice agent in dev mode..."
	@docker compose run --rm voice-agent python -m src.voice_agent dev

docker-ps:                               ## Show container status
	@docker compose ps

all:                                     ## Start everything (LiveKit + web app)
	@echo "Starting full app..."
	@echo "Starting LiveKit server + voice agent in Docker..."
	@docker compose up -d --build
	@echo ""
	@echo "Starting web app (foreground)..."
	@echo "Press Ctrl+C to stop web app"
	@echo "  (Docker containers keep running - use 'make docker-down' to stop)"
	@echo ""
	@cd web && $(PNPM) dev
