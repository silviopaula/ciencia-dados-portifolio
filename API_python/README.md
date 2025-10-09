# API de Usuários - FastAPI + Pandas

API REST desenvolvida com FastAPI para análise e consulta de dados de usuários, com processamento de dados utilizando Pandas.

## Descrição

Este projeto demonstra a construção de uma API completa para análise de dados de usuários, incluindo processamento de dados aninhados em JSON, cálculos estatísticos e endpoints otimizados para consultas específicas.

## Tecnologias

- **Python 3.10.18+**
- **FastAPI** - Framework web moderno e de alta performance
- **Pandas** - Análise e manipulação de dados
- **Uvicorn** - Servidor ASGI
- **Pydantic** - Validação de dados

## 📦 Instalação

### Pré-requisitos

- Python 3.10.18 ou superior
- pip

1. Instale as dependências
```bash
pip install -r requirements.txt
```

## Endpoints

### `GET /superusers`
Retorna usuários com score >= 900 e status ativo.

### `GET /top-countries`
Retorna os 5 países com maior número de superusuários.

### `GET /team-insights`
Retorna estatísticas detalhadas por equipe.

### `GET /active-users-per-day`
Retorna contagem de logins por dia.

### `GET /evaluation`
Executa autoavaliação da API e retorna relatório de desempenho.

## Uso
1. Acesse a documentação interativa
```
http://127.0.0.1:8000/docs
```

2. Teste os endpoints via curl
```bash
curl http://127.0.0.1:8000/superusers
curl http://127.0.0.1:8000/top-countries
curl http://127.0.0.1:8000/team-insights
curl http://127.0.0.1:8000/active-users-per-day?min=3000
curl http://127.0.0.1:8000/evaluation
```
## 📝 Licença

Este projeto está sob a licença MIT.
---
