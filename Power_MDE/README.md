# Power e MDE em Difference-in-Differences

Notebook curto em R que simula um painel simples e mostra, de forma prática, como calcular e interpretar o poder estatístico (power) e o Minimum Detectable Effect (MDE) em um exemplo TWFE.

## Por que isso importa
Seu modelo não detectou impacto ou não tinha sensibilidade para enxergar?
Em avaliações de impacto causal é comum encontrar resultados “não significativos”. Na prática, muita gente interpreta isso como: “não houve efeito” mas essa conclusão pode ser precipitada.
Resultados não significativos podem significar simplesmente que o estudo não tinha sensibilidade suficiente para enxergar o impacto que realmente existia. Esse é o ponto crítico onde entram o Poder Estatístico (Power) e o Minimum Detectable Effect (MDE).

Em termos simples:
- Power é a chance de detectar um efeito verdadeiro.
- MDE é o menor impacto que o estudo consegue perceber com confiança.

Se o MDE for maior que o efeito esperado, o estudo fica “míope”: o impacto pode estar acontecendo, mas os dados não têm resolução suficiente para enxergá-lo. Isso é especialmente comum em bases de dados pequenas ou políticas que geram efeitos modestos, porém relevantes.

Realizei um simples exercício para demostrar como se calcula o Power e o Minimum Detectable Effect (MDE). Em meu exemplo aplicado com Difference-in-Differences TWFE, o erro-padrão indicava que o estudo só conseguia detectar efeitos acima de aproximadamente 0.25. Qualquer efeito menor que isso simplesmente não teria chance de aparecer como significativo, ainda que fosse real.

A mensagem central é simples e importante:
Ausência de significância não significa ausência de impacto.
Muitas vezes, significa apenas ausência de poder estatístico.

Para quem trabalha com dados, decisões e experimentos, entender power e o Minimum Detectable Effect (MDE), não é um detalhe técnico. É um requisito fundamental para evitar conclusões equivocadas, proteger investimentos e tomar decisões mais sólidas, baseadas na sensibilidade real do estudo.

## Como ver o exemplo
Abra e rode o notebook `Jupyter R - Calculado o Power e o MDE.ipynb` no Jupyter ou RStudio para reproduzir o exercício.