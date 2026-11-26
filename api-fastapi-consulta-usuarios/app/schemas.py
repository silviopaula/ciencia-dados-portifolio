"""Modelos de entrada e saída (Pydantic v2) da API de Usuários.

Define contratos tipados que alimentam automaticamente a validação
de dados e a documentação interativa OpenAPI/Swagger (/docs).
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# 1. SCHEMAS DE ENTRADA (USUÁRIOS E ESTRUTURAS ANINHADAS)
# =============================================================================
class ProjectSchema(BaseModel):
    name: str = Field(..., description="Nome do projeto atribuído à equipe")
    completed: bool = Field(..., description="Indica se o projeto foi concluído")


class TeamSchema(BaseModel):
    name: str = Field(..., description="Nome da equipe")
    leader: bool = Field(..., description="Indica se o usuário é líder da equipe")
    projects: List[ProjectSchema] = Field(default_factory=list, description="Projetos da equipe")


class LogSchema(BaseModel):
    date: str = Field(..., description="Data do evento de log (YYYY-MM-DD)")
    action: str = Field(..., description="Ação registrada (ex.: login, logout)")


class UserInSchema(BaseModel):
    id: str = Field(..., description="Identificador único (UUID) do usuário")
    name: str = Field(..., description="Nome completo do usuário")
    age: int = Field(..., ge=0, description="Idade do usuário")
    score: int = Field(..., ge=0, description="Pontuação do usuário")
    active: bool = Field(..., description="Status de atividade do usuário")
    country: str = Field(..., description="País do usuário")
    team: TeamSchema
    logs: List[LogSchema] = Field(default_factory=list)


# =============================================================================
# 2. SCHEMAS DE RESPOSTA DOS ENDPOINTS ANALÍTICOS
# =============================================================================
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    users_loaded: int
    docs_url: str = "/docs"


class LoadUsersResponse(BaseModel):
    message: str
    user_count: int
    processing_time_ms: float


class SuperuserItem(BaseModel):
    id: str
    name: str
    age: int
    score: int
    active: bool
    country: str
    team: str
    is_leader: bool


class SuperusersResponse(BaseModel):
    timestamp: str
    processing_time_ms: float
    total: int
    returned: int
    limit: Optional[int] = None
    offset: int = 0
    data: List[SuperuserItem]


class CountryCountItem(BaseModel):
    country: str
    total: int


class TopCountriesResponse(BaseModel):
    timestamp: str
    processing_time_ms: float
    top_n: int
    data: List[CountryCountItem]


class TeamInsightItem(BaseModel):
    team: str
    total_members: int
    leaders: int
    completed_projects: int
    active_percentage: float


class TeamInsightsResponse(BaseModel):
    timestamp: str
    processing_time_ms: float
    total_teams: int
    data: List[TeamInsightItem]


class DailyLoginsItem(BaseModel):
    date: str
    total: int


class ActiveUsersPerDayResponse(BaseModel):
    timestamp: str
    processing_time_ms: float
    min_filter: Optional[int] = None
    total_days: int
    data: List[DailyLoginsItem]


class EndpointEvaluationItem(BaseModel):
    endpoint: str
    status_code: int
    response_time_ms: float
    is_valid_json: bool
    passed: bool


class EvaluationResponse(BaseModel):
    timestamp: str
    processing_time_ms: float
    score_percentage: float
    passed: int
    total: int
    results: List[EndpointEvaluationItem]
