"""Camada de serviços e armazenamento em memória da API de Usuários.

Responsável por carregar o arquivo JSON aninhado (ex.: 100.000 usuários),
manter os dados em memória e executar as agregações analíticas com alta
performance (respostas na casa de poucos milissegundos).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "usuarios.json"


def utc_now_iso() -> str:
    """Retorna o timestamp atual em formato ISO-8601 (UTC)."""
    return datetime.now(timezone.utc).isoformat()


class UserAnalyticsRepository:
    """Repositório em memória com índices analíticos pré-computados.

    Ao carregar os usuários uma única vez (na inicialização ou via POST /users),
    mantemos a lista bruta e estruturas leves em memória para que consultas
    repetidas tenham latência mínima.
    """

    def __init__(self) -> None:
        self._users: List[Dict[str, Any]] = []
        self._superusers: List[Dict[str, Any]] = []
        self._top_countries: List[Dict[str, Any]] = []
        self._team_insights: List[Dict[str, Any]] = []
        self._daily_logins: List[Dict[str, Any]] = []

    @property
    def user_count(self) -> int:
        return len(self._users)

    def load_from_file(self, file_path: Path = DEFAULT_DATA_PATH) -> int:
        """Carrega os usuários a partir de um arquivo JSON no disco."""
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            raw_users = json.load(f)

        return self.set_users(raw_users)

    def set_users(self, users: List[Dict[str, Any]]) -> int:
        """Armazena a lista de usuários em memória e atualiza os índices analíticos."""
        self._users = users
        self._rebuild_analytics()
        return len(self._users)

    def _rebuild_analytics(self) -> None:
        """Pré-processa superusuários, rankings, times e logins diários em passagem única O(N)."""
        superusers: List[Dict[str, Any]] = []
        country_counter: Counter[str] = Counter()
        daily_login_counter: Counter[str] = Counter()

        # Estatísticas por equipe
        team_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {
                "total_members": 0,
                "leaders": 0,
                "completed_projects": 0,
                "active_members": 0,
            }
        )

        for user in self._users:
            is_active = bool(user.get("active", False))
            score = int(user.get("score", 0))
            country = str(user.get("country", "Desconhecido"))
            team = user.get("team") or {}
            team_name = str(team.get("name", "Sem Equipe"))
            is_leader = bool(team.get("leader", False))
            projects = team.get("projects") or []

            # 1. Filtro de Superusuários (score >= 900 e ativo)
            if score >= 900 and is_active:
                superusers.append(
                    {
                        "id": str(user.get("id", "")),
                        "name": str(user.get("name", "")),
                        "age": int(user.get("age", 0)),
                        "score": score,
                        "active": is_active,
                        "country": country,
                        "team": team_name,
                        "is_leader": is_leader,
                    }
                )
                country_counter[country] += 1

            # 2. Acumulação de métricas por equipe
            t_entry = team_stats[team_name]
            t_entry["total_members"] += 1
            if is_leader:
                t_entry["leaders"] += 1
            if is_active:
                t_entry["active_members"] += 1
            for proj in projects:
                if proj.get("completed"):
                    t_entry["completed_projects"] += 1

            # 3. Contagem de logins diários nos logs aninhados
            for log in user.get("logs") or []:
                if log.get("action") == "login" and log.get("date"):
                    daily_login_counter[str(log["date"])] += 1

        # Salva resultados ordenados para consulta O(1) / O(K) nos endpoints
        self._superusers = superusers

        self._top_countries = [
            {"country": country, "total": count}
            for country, count in country_counter.most_common()
        ]

        self._team_insights = sorted(
            [
                {
                    "team": team_name,
                    "total_members": stats["total_members"],
                    "leaders": stats["leaders"],
                    "completed_projects": stats["completed_projects"],
                    "active_percentage": round(
                        (stats["active_members"] / stats["total_members"]) * 100.0, 2
                    )
                    if stats["total_members"] > 0
                    else 0.0,
                }
                for team_name, stats in team_stats.items()
            ],
            key=lambda item: item["total_members"],
            reverse=True,
        )

        self._daily_logins = [
            {"date": date_str, "total": count}
            for date_str, count in sorted(daily_login_counter.items())
        ]

    # -------------------------------------------------------------------------
    # Métodos de consulta usados pelos endpoints
    # -------------------------------------------------------------------------
    def get_superusers(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> tuple[int, List[Dict[str, Any]]]:
        total = len(self._superusers)
        if limit is None:
            sliced = self._superusers[offset:]
        else:
            sliced = self._superusers[offset : offset + limit]
        return total, sliced

    def get_top_countries(self, top_n: int = 5) -> List[Dict[str, Any]]:
        return self._top_countries[:top_n]

    def get_team_insights(self) -> List[Dict[str, Any]]:
        return self._team_insights

    def get_active_users_per_day(self, min_logins: Optional[int] = None) -> List[Dict[str, Any]]:
        if min_logins is None:
            return self._daily_logins
        return [item for item in self._daily_logins if item["total"] >= min_logins]


# Instância singleton compartilhada pela aplicação FastAPI
repository = UserAnalyticsRepository()
