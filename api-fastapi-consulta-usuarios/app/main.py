"""Aplicação principal FastAPI — API de Análise de Usuários.

Para executar localmente:
    uvicorn app.main:app --reload

Documentação interativa (Swagger UI):
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
import httpx

from app.schemas import (
    ActiveUsersPerDayResponse,
    EndpointEvaluationItem,
    EvaluationResponse,
    HealthResponse,
    LoadUsersResponse,
    SuperusersResponse,
    TeamInsightsResponse,
    TopCountriesResponse,
    UserInSchema,
)
from app.services import DEFAULT_DATA_PATH, repository, utc_now_iso


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Carrega automaticamente o arquivo data/usuarios.json na inicialização, se existir."""
    if DEFAULT_DATA_PATH.exists() and repository.user_count == 0:
        repository.load_from_file(DEFAULT_DATA_PATH)
    yield


app = FastAPI(
    title="API de Análise de Usuários — FastAPI",
    description=(
        "API REST de alta performance construída com FastAPI e Pydantic v2 para "
        "ingestão e análise de dados aninhados de usuários (equipes, projetos e logs de acesso)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def _require_loaded_data() -> None:
    if repository.user_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Nenhum usuário carregado em memória. Envie dados via POST /users primeiro.",
        )


@app.get("/", response_model=HealthResponse, tags=["Status"])
def health_check() -> HealthResponse:
    """Verifica o status da API e a quantidade de usuários atualmente carregados em memória."""
    return HealthResponse(
        status="online",
        service="API de Análise de Usuários",
        version=app.version,
        users_loaded=repository.user_count,
    )


@app.post("/users", response_model=LoadUsersResponse, status_code=201, tags=["Ingestão"])
def upload_users(users: List[UserInSchema]) -> LoadUsersResponse:
    """Recebe uma lista de usuários em JSON, valida o contrato via Pydantic e atualiza o repositório em memória."""
    start = time.perf_counter()
    raw_payload = [u.model_dump() for u in users]
    count = repository.set_users(raw_payload)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return LoadUsersResponse(
        message="Base de usuários carregada e indexada com sucesso.",
        user_count=count,
        processing_time_ms=elapsed_ms,
    )


@app.get("/superusers", response_model=SuperusersResponse, tags=["Analytics"])
def get_superusers(
    limit: Optional[int] = Query(
        default=100,
        ge=1,
        le=50000,
        description="Quantidade máxima de registros retornados (use paginação para payloads leves).",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Deslocamento inicial para paginação dos superusuários.",
    ),
) -> SuperusersResponse:
    """Retorna superusuários (score >= 900 e active == True) com suporte a paginação."""
    _require_loaded_data()
    start = time.perf_counter()
    total, items = repository.get_superusers(limit=limit, offset=offset)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return SuperusersResponse(
        timestamp=utc_now_iso(),
        processing_time_ms=elapsed_ms,
        total=total,
        returned=len(items),
        limit=limit,
        offset=offset,
        data=items,
    )


@app.get("/top-countries", response_model=TopCountriesResponse, tags=["Analytics"])
def get_top_countries(
    top_n: int = Query(
        default=5,
        ge=1,
        le=50,
        description="Número de países no ranking de superusuários.",
    ),
) -> TopCountriesResponse:
    """Retorna os países com maior concentração de superusuários."""
    _require_loaded_data()
    start = time.perf_counter()
    items = repository.get_top_countries(top_n=top_n)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return TopCountriesResponse(
        timestamp=utc_now_iso(),
        processing_time_ms=elapsed_ms,
        top_n=top_n,
        data=items,
    )


@app.get("/team-insights", response_model=TeamInsightsResponse, tags=["Analytics"])
def get_team_insights() -> TeamInsightsResponse:
    """Retorna métricas agregadas por equipe: membros, líderes, projetos concluídos e % de ativos."""
    _require_loaded_data()
    start = time.perf_counter()
    items = repository.get_team_insights()
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return TeamInsightsResponse(
        timestamp=utc_now_iso(),
        processing_time_ms=elapsed_ms,
        total_teams=len(items),
        data=items,
    )


@app.get("/active-users-per-day", response_model=ActiveUsersPerDayResponse, tags=["Analytics"])
def get_active_users_per_day(
    min: Optional[int] = Query(
        default=None,
        ge=0,
        description="Filtra apenas os dias com total de logins maior ou igual a este limite.",
    ),
) -> ActiveUsersPerDayResponse:
    """Retorna a contagem diária de eventos de login, com filtro opcional (?min=)."""
    _require_loaded_data()
    start = time.perf_counter()
    items = repository.get_active_users_per_day(min_logins=min)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return ActiveUsersPerDayResponse(
        timestamp=utc_now_iso(),
        processing_time_ms=elapsed_ms,
        min_filter=min,
        total_days=len(items),
        data=items,
    )


@app.get("/evaluation", response_model=EvaluationResponse, tags=["Observabilidade"])
async def evaluate_api() -> EvaluationResponse:
    """Executa autoavaliação interna (ASGI in-process) de todos os endpoints analíticos.

    Verifica status HTTP 200, latência em milissegundos e validade do payload JSON
    sem risco de deadlock de rede.
    """
    _require_loaded_data()
    overall_start = time.perf_counter()

    endpoints_to_test = [
        "/superusers?limit=10",
        "/top-countries",
        "/team-insights",
        "/active-users-per-day",
        "/active-users-per-day?min=3000",
    ]

    results: List[EndpointEvaluationItem] = []
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://internal-eval") as client:
        for ep in endpoints_to_test:
            t0 = time.perf_counter()
            resp = await client.get(ep)
            dt_ms = round((time.perf_counter() - t0) * 1000, 2)

            is_valid_json = False
            try:
                resp.json()
                is_valid_json = True
            except Exception:
                is_valid_json = False

            passed = (resp.status_code == 200) and is_valid_json
            results.append(
                EndpointEvaluationItem(
                    endpoint=ep,
                    status_code=resp.status_code,
                    response_time_ms=dt_ms,
                    is_valid_json=is_valid_json,
                    passed=passed,
                )
            )

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    score_pct = round((passed_count / total_count) * 100.0, 2) if total_count else 0.0
    total_ms = round((time.perf_counter() - overall_start) * 1000, 2)

    return EvaluationResponse(
        timestamp=utc_now_iso(),
        processing_time_ms=total_ms,
        score_percentage=score_pct,
        passed=passed_count,
        total=total_count,
        results=results,
    )
