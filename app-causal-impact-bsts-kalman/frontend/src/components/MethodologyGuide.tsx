import React from "react";
import { BookOpenCheck, CheckCircle2, FlaskConical, GitCompare, ShieldCheck, Sigma } from "lucide-react";

const PILLARS = [
  {
    icon: GitCompare,
    title: "1. O Contrafactual em Séries Temporais",
    color: "text-cyan-300 bg-cyan-500/15 ring-cyan-500/30",
    body: "Quando uma campanha, política pública ou mudança de preço é lançada sem um experimento A/B randomizado, não observamos o que teria acontecido sem a intervenção. O CausalImpact constrói essa trajetória contrafactual combinando a dinâmica histórica de Y com séries de controle correlacionadas.",
  },
  {
    icon: Sigma,
    title: "2. Modelo Estrutural em Espaço de Estados (BSTS)",
    color: "text-violet-300 bg-violet-500/15 ring-violet-500/30",
    body: "Em vez de uma regressão estática simples, o modelo decompõe Y_t em nível latente (Random Walk ou tendência linear local), sazonalidade estocástica opcional e combinação linear das covariáveis de controle X_tβ, ajustados exclusivamente no período pré-intervenção via Filtro/Suavizador de Kalman.",
  },
  {
    icon: ShieldCheck,
    title: "3. Hipótese Crítica: No-Spillover nos Controles",
    color: "text-emerald-300 bg-emerald-500/15 ring-emerald-500/30",
    body: "A validade causal exige que as séries explicativas (X) NÃO tenham sido afetadas pela intervenção (ausência de contaminação / spillover) e que sua relação com Y permaneça estável. Por isso, este app inclui auditoria automática de quebra estrutural em cada controle.",
  },
  {
    icon: FlaskConical,
    title: "4. Falsificação via Teste Placebo no Tempo",
    color: "text-amber-300 bg-amber-500/15 ring-amber-500/30",
    body: "Antes de confiar no impacto estimado, simulamos uma intervenção fictícia dentro do período pré-intervenção (onde sabemos que o efeito verdadeiro é zero). Se o placebo não acusar falso impacto (p ≥ α), ganhamos forte evidência de que o contrafactual está bem especificado.",
  },
];

export const MethodologyGuide: React.FC = () => {
  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 via-slate-900/80 to-cyan-950/20 p-6">
        <div className="flex items-start gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-cyan-500/15 text-cyan-300 ring-1 ring-cyan-500/30">
            <BookOpenCheck className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-extrabold text-white">
              Guia Metodológico — Inferência Causal com Modelos Estruturais (Brodersen et al., 2015)
            </h2>
            <p className="mt-1 text-sm leading-relaxed text-slate-300">
              Implementação nativa em Python (<code className="text-cyan-300">Statsmodels UnobservedComponents</code> +
              Filtro de Kalman + Simulação Preditiva Posterior Conjunta Monte Carlo), projetada para rigor estatístico,
              zero data leakage e interpretação executiva imediata.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {PILLARS.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.title} className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
              <div className="flex items-center gap-3">
                <span className={`flex h-9 w-9 items-center justify-center rounded-xl ring-1 ${item.color}`}>
                  <Icon className="h-4 w-4" />
                </span>
                <h3 className="text-sm font-bold text-white">{item.title}</h3>
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-300">{item.body}</p>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
          <h3 className="mb-3 text-sm font-bold text-white">Qual Especificação Estrutural Escolher?</h3>
          <ul className="space-y-2.5 text-xs leading-relaxed text-slate-300">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-cyan-400" />
              <span>
                <strong className="text-white">Local Level (Padrão Google CausalImpact):</strong> melhor equilíbrio
                entre flexibilidade e estabilidade quando existem boas covariáveis de controle. Evita extrapolações
                explosivas de tendência no horizonte pós-intervenção.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
              <span>
                <strong className="text-white">Local Linear Trend:</strong> útil quando Y possui inclinação própria
                persistente não capturada integralmente pelas covariáveis (gera bandas mais largas em horizontes longos).
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
              <span>
                <strong className="text-white">Local Level + Sazonalidade Estocástica:</strong> incorpora o ciclo sazonal
                (ex.: 12 meses ou 7 dias) diretamente na matriz de estados, preferível a pré-desazonalizar via STL pois
                propaga a incerteza sazonal para os intervalos de credibilidade.
              </span>
            </li>
          </ul>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-5">
          <h3 className="mb-3 text-sm font-bold text-white">Referências Acadêmicas</h3>
          <div className="space-y-3 text-xs leading-relaxed text-slate-300">
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
              <p className="font-semibold text-cyan-300">
                Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015)
              </p>
              <p className="mt-1 italic text-slate-400">
                Inferring causal impact using Bayesian structural time-series models. The Annals of Applied Statistics,
                9(1), 247–274.
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3.5">
              <p className="font-semibold text-emerald-300">Durbin, J., & Koopman, S. J. (2012)</p>
              <p className="mt-1 italic text-slate-400">
                Time Series Analysis by State Space Methods (2nd ed.). Oxford University Press.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
