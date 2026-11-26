-- ============================================================================
-- Extração Upstream — Google BigQuery (Base dos Dados / TSE)
-- Alimenta a camada Bronze no Databricks Unity Catalog:
--   workspace.avaliacao_lei_tse.candidatos
--   workspace.avaliacao_lei_tse.candidatos_eleitos
-- ============================================================================

-- 1) Tabela Bronze: Candidaturas registradas no TSE (1992–2022)
SELECT
    CAST(ano AS INT64) AS ano,
    UPPER(TRIM(sigla_uf)) AS sigla_uf,
    UPPER(TRIM(sigla_uf_nascimento)) AS sigla_uf_nascimento,
    TRIM(titulo_eleitoral) AS titulo_eleitoral,
    TRIM(nome) AS nome,
    LOWER(TRIM(genero)) AS genero,
    LOWER(TRIM(cargo)) AS cargo,
    LOWER(TRIM(situacao)) AS situacao,
    sigla_partido,
    instrucao,
    idade
FROM `basedosdados.br_tse_eleicoes.candidatos`
WHERE ano BETWEEN 1992 AND 2022
  AND cargo IS NOT NULL;


-- 2) Tabela Bronze: Candidatos Eleitos por UF, Ano e Cargo (1992–2022)
SELECT
    CAST(ano AS INT64) AS ano,
    UPPER(TRIM(sigla_uf)) AS sigla_uf,
    LOWER(TRIM(cargo)) AS cargo,
    TRIM(titulo_eleitoral_candidato) AS titulo_eleitoral_candidato,
    LOWER(TRIM(resultado)) AS resultado,
    SUM(COALESCE(votos, 0)) AS votos
FROM `basedosdados.br_tse_eleicoes.resultados_candidato`
WHERE ano BETWEEN 1992 AND 2022
  AND REGEXP_CONTAINS(LOWER(resultado), r'^eleit')
GROUP BY
    ano,
    sigla_uf,
    cargo,
    titulo_eleitoral_candidato,
    resultado;
