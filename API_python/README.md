# ⚡ API REST de Análise de Usuários — FastAPI + Pydantic v2

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-4B0082?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)

**API REST modular de alta performance para ingestão, indexação em memória e consultas analíticas sobre 100.000 usuários em JSON aninhado (~67 MB)**

</div>

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Arquitetura e Decisões Técnicas](#-arquitetura-e-decisões-técnicas)
- [Endpoints Disponíveis](#-endpoints-disponíveis)
- [Performance e Resultados (100.000 Usuários)](#-performance-e-resultados-100000-usuários)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar e Testar](#-como-executar-e-testar)
- [Exemplos de Consumo (curl e Python)](#-exemplos-de-consumo-curl-e-python)
- [Autor](#-autor)

---

## 🎯 Sobre o Projeto

Este projeto implementa uma **API REST profissional em Python com FastAPI**, projetada para lidar com grandes volumes de dados JSON aninhados (`data/usuarios.json`, contendo **100.000 usuários**, equipes, projetos e históricos de logs de login/logout — aproximadamente **67 MB**).

A aplicação separa de forma limpa os contratos de dados (**Pydantic v2**), a camada de processamento e indexação em memória (**Services**), as rotas HTTP (**FastAPI**) e os **testes automatizados**, acompanhados de um **Jupyter Notebook didático** que demonstra o consumo da API e visualiza os insights retornados.

---

## 🏗️ Arquitetura e Decisões Técnicas

1. **Separação de Responsabilidades (`app/`)**:
   - `app/schemas.py`: Modelos de entrada e saída fortemente tipados (Pydantic v2), gerando documentação interativa automática via **Swagger UI** (`/docs`) e **ReDoc** (`/redoc`).
   - `app/services.py`: Classe `UserAnalyticsRepository` que processa os 100.000 registros aninhados em **uma única passagem linear $O(N)$**, pré-computando estruturas leves em memória.
   - `app/main.py`: Aplicação FastAPI usando o gerenciador moderno de ciclo de vida (`lifespan`) para carregar automaticamente `data/usuarios.json` na inicialização.
2. **Latência Sub-Milissegundo**:
   - A carga e indexação completa do arquivo de 67 MB leva **~0,65 s** na inicialização.
   - Após carregado, todos os endpoints analíticos respondem em **menos de 2 ms**.
3. **Paginação Segura e Observabilidade**:
   - O endpoint `/superusers` suporta `limit` e `offset` para evitar trafegar mais de 10.000 objetos JSON de uma só vez no navegador.
   - O endpoint `/evaluation` executa uma autoavaliação interna assíncrona via `httpx.ASGITransport` (sem deadlock de rede), verificando status HTTP, latência e integridade JSON.

---

## 🔗 Endpoints Disponíveis

| Método | Endpoint | Descrição | Parâmetros Opcionais |
| :---: | :--- | :--- | :--- |
| `GET` | `/` | *Health check*, versão e total de usuários em memória | — |
| `POST` | `/users` | Ingestão de nova lista de usuários JSON (valida via Pydantic e reindexa) | Body: `List[UserInSchema]` |
| `GET` | `/superusers` | Lista usuários com `score >= 900` e `active == True` | `?limit=100&offset=0` |
| `GET` | `/top-countries` | Ranking dos países com mais superusuários | `?top_n=5` |
| `GET` | `/team-insights` | Membros, líderes, projetos concluídos e `%` de ativos por equipe | — |
| `GET` | `/active-users-per-day` | Total de eventos de `login` por data | `?min=34300` |
| `GET` | `/evaluation` | Auditoria automática de status, JSON e latência dos endpoints | — |

---

## 📊 Performance e Resultados (100.000 Usuários)

Resultados reais obtidos sobre `data/usuarios.json` (verificados no notebook [`demo_consumo_api.ipynb`](demo_consumo_api.ipynb)):

- **Inicialização + Indexação em Memória (67 MB)**: **`0,65 segundos`**
- **Total de Usuários Carregados**: **`100.000`**
- **Total de Superusuários (`score >= 900` e ativo)**: **`10.132`**
- **Top 5 Países em Superusuários**: Canadá (`1.057`), França (`1.050`), Índia (`1.042`), EUA (`1.042`) e México (`1.020`).
- **Equipes Analisadas**: 5 equipes (`Fullstack Force`, `UX Wizards`, `Frontend Avengers`, `DevOps Masters`, `Backend Ninjas`), todas com ~90% de membros ativos e ~28.000 projetos concluídos.

### Relatório do Endpoint `GET /evaluation`

| Endpoint Auditado | Status HTTP | Latência (ms) | JSON Válido | Resultado |
| :--- | :---: | :---: | :---: | :---: |
| `/superusers?limit=10` | `200` | `~1,56 ms` | Sim | ✅ Aprovado |
| `/top-countries` | `200` | `~0,92 ms` | Sim | ✅ Aprovado |
| `/team-insights` | `200` | `~0,78 ms` | Sim | ✅ Aprovado |
| `/active-users-per-day` | `200` | `~0,65 ms` | Sim | ✅ Aprovado |
| `/active-users-per-day?min=3000` | `200` | `~0,66 ms` | Sim | ✅ Aprovado |

---

## 📁 Estrutura do Projeto

```text
API_python/
├── app/
│   ├── __init__.py           # Inicialização do pacote
│   ├── main.py               # Aplicação FastAPI, ciclo de vida (lifespan) e rotas
│   ├── schemas.py            # Contratos de entrada/saída com Pydantic v2
│   └── services.py           # Repositório em memória e agregações O(N)
├── tests/
│   └── test_api.py           # Suíte de 8 testes automatizados (unitários + integração)
├── data/
│   └── usuarios.json         # Base de 100.000 usuários em JSON aninhado (~67 MB)
├── demo_consumo_api.ipynb    # Notebook didático consumindo todos os endpoints + gráficos
├── demo_consumo_api.html     # Versão renderizada em HTML do notebook de demonstração
├── requirements.txt          # Dependências Python limpas
└── README.md                 # Documentação do projeto
```

---

## 🚀 Como Executar e Testar

1. **Acesse a pasta do projeto e instale as dependências:**
   ```bash
   cd API_python
   pip install -r requirements.txt
   ```

2. **Suba o servidor da API com Uvicorn:**
   ```bash
   uvicorn app.main:app --reload
   ```
   - API rodando em: `http://127.0.0.1:8000`
   - Documentação interativa (Swagger UI): `http://127.0.0.1:8000/docs`

3. **Execute a suíte de testes automatizados:**
   ```bash
   python -m unittest discover -s tests -v
   # ou com pytest:
   pytest -v
   ```

---

## 💻 Exemplos de Consumo (`curl` e Python)

### Via `curl` (Terminal)
```bash
curl http://127.0.0.1:8000/
curl "http://127.0.0.1:8000/superusers?limit=5&offset=0"
curl http://127.0.0.1:8000/top-countries
curl http://127.0.0.1:8000/team-insights
curl "http://127.0.0.1:8000/active-users-per-day?min=34300"
curl http://127.0.0.1:8000/evaluation
```

### Via Python (`requests`)
```python
import requests

resp = requests.get("http://127.0.0.1:8000/top-countries", params={"top_n": 5})
print(resp.json())
```

---

## 👤 Autor

**Silvio da Rosa Paula**  
*Doutor em Economia Aplicada (UFPel) | Especialista em Inteligência Artificial e Ciência de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/silvio-paula/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/silviopaula)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:silvio.rosa.paula@gmail.com)
