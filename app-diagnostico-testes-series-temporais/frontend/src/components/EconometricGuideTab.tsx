import React from "react";
import { BookOpen, GitFork, ShieldAlert } from "lucide-react";

const TEST_CARDS = [
  {
    group: "Estacionariedade & Raiz Unitária",
    name: "Dickey-Fuller Aumentado (ADF)",
    h0: "Existe raiz unitária (série NÃO é estacionária).",
    rule: "p < α → Rejeita H₀ (evidência de estacionariedade).",
    detail:
      "Adiciona defasagens da 1ª diferença selecionadas por AIC para limpar autocorrelação serial nos erros. Avaliado em três especificações: sem constante (n), constante (c) e constante + tendência (ct).",
  },
  {
    group: "Estacionariedade & Raiz Unitária",
    name: "Phillips-Perron (PP)",
    h0: "Existe raiz unitária (série NÃO é estacionária).",
    rule: "p < α → Rejeita H₀ (série estacionária).",
    detail:
      "Corrige não-parametricamente a estatística t via variância de longo prazo de Newey-West (núcleo de Bartlett), sendo robusto a autocorrelação de ordem superior e heterocedasticidade.",
  },
  {
    group: "Estacionariedade & Raiz Unitária",
    name: "KPSS (Kwiatkowski-Phillips-Schmidt-Shin)",
    h0: "A série É estacionária em nível (c) ou em torno de tendência (ct).",
    rule: "p ≥ α → Não rejeita H₀ (série consistente com estacionariedade).",
    detail:
      "Possui hipótese nula invertida em relação ao ADF/PP. Combinar ADF/PP com KPSS elimina conclusões precipitadas causadas por baixo poder estatístico em amostras finitas.",
  },
  {
    group: "Quebra Estrutural Endógena",
    name: "Zivot-Andrews (1992)",
    h0: "Existe raiz unitária mesmo permitindo uma quebra estrutural endógena.",
    rule: "p < α → Série estacionária em torno de quebra na data identificada.",
    detail:
      "Corrige o célebre viés de Perron (1989): choques abruptos de nível ou inclinação (como crises de 2008/2020) enganam o ADF tradicional fazendo séries estacionárias parecerem I(1).",
  },
  {
    group: "Autocorrelação Residual",
    name: "Ljung-Box (Q*) & Box-Pierce (Q)",
    h0: "Os resíduos são independentes (Ruído Branco até o lag m).",
    rule: "p ≥ α → Resíduos limpos; o modelo capturou a dinâmica temporal.",
    detail:
      "Ljung-Box corrige o viés de amostra finita de Box-Pierce. Nos resíduos de modelos SARIMA, ajustamos os graus de liberdade por model_df = p + q + P + Q para evitar distorção do p-valor.",
  },
  {
    group: "Heterocedasticidade & Volatilidade",
    name: "Engle ARCH-LM & Breusch-Pagan Temporal",
    h0: "Variância condicional (ARCH) ou incondicional no tempo (BP) é constante.",
    rule: "p ≥ α → Homocedasticidade; p < α → Volatilidade variável no tempo.",
    detail:
      "ARCH-LM detecta clusters de volatilidade (típicos de séries financeiras → modelos GARCH). Breusch-Pagan Temporal detecta variância que cresce com o nível (sinal clássico para aplicar Log ou Box-Cox).",
  },
  {
    group: "Normalidade & Caudas Pesadas",
    name: "Anderson-Darling, Jarque-Bera & Shapiro-Wilk",
    h0: "Os resíduos seguem distribuição Normal.",
    rule: "p ≥ α → Resíduos Gaussianos (ICs paramétricos confiáveis).",
    detail:
      "Anderson-Darling dá peso extra às caudas; Jarque-Bera decompõe o desvio em Assimetria e Excesso de Curtose; Shapiro-Wilk tem maior poder estatístico em amostras pequenas e médias.",
  },
];

export const EconometricGuideTab: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Matriz 2x2 ADF/PP x KPSS */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-indigo-600" />
            Matriz de Decisão Cruzada: ADF / Phillips-Perron × KPSS
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Nunca confie em um único teste de raiz unitária: o cruzamento das hipóteses nulas opostas identifica o regime econométrico real da série.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 p-3.5">
            <div className="font-bold text-emerald-900">1. ADF/PP rejeitam H₀ + KPSS não rejeita H₀</div>
            <p className="text-emerald-800 mt-1">
              <strong>Consenso Forte de Estacionariedade — I(0):</strong> a série flutua em torno de média estável. Pode ajustar modelos ARMA/SARIMA diretamente com <code>d = 0</code>.
            </p>
          </div>
          <div className="rounded-lg border border-rose-200 bg-rose-50/70 p-3.5">
            <div className="font-bold text-rose-900">2. ADF/PP não rejeitam H₀ + KPSS rejeita H₀</div>
            <p className="text-rose-800 mt-1">
              <strong>Raiz Unitária Confirmada — I(1):</strong> aplique 1ª diferença (<code>Δyₜ</code>) ou log-retorno (<code>Δ ln yₜ</code>) e reavalie no Laboratório de Transformações.
            </p>
          </div>
          <div className="rounded-lg border border-amber-200 bg-amber-50/70 p-3.5">
            <div className="font-bold text-amber-900">3. ADF/PP não rejeitam H₀ + Zivot-Andrews rejeita H₀</div>
            <p className="text-amber-800 mt-1">
              <strong>Viés de Perron (Quebra Estrutural):</strong> um choque abrupto de nível/tendência mascarou a estacionariedade. Inclua variável dummy de intervenção ou diferencie.
            </p>
          </div>
          <div className="rounded-lg border border-indigo-200 bg-indigo-50/70 p-3.5">
            <div className="font-bold text-indigo-900">4. Sinais Divergentes (ADF ≠ KPSS)</div>
            <p className="text-indigo-800 mt-1">
              <strong>Tendência Determinística / Sazonalidade / Integração Fracionária:</strong> inspecione a decomposição STL, teste <code>Δₛyₜ</code> (diferença sazonal) e verifique a queda da ACF.
            </p>
          </div>
        </div>
      </div>

      {/* Fichas Técnicas dos Testes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {TEST_CARDS.map((card) => (
          <div key={card.name} className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600">
              {card.group}
            </span>
            <h4 className="text-sm font-bold text-slate-900">{card.name}</h4>
            <p className="text-xs text-slate-700">
              <strong>H₀:</strong> {card.h0}
            </p>
            <p className="text-xs text-emerald-800 bg-emerald-50 rounded px-2.5 py-1.5 font-medium">
              {card.rule}
            </p>
            <p className="text-xs text-slate-600 leading-relaxed">{card.detail}</p>
          </div>
        ))}
      </div>

      {/* Box-Jenkins & Árvore de Decisão Prática */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <GitFork className="w-4 h-4 text-indigo-600" />
            Identificação Box-Jenkins via ACF & PACF
          </h3>
          <table className="w-full text-xs text-left border border-slate-200 rounded-lg overflow-hidden">
            <thead className="bg-slate-50 text-slate-700 font-semibold">
              <tr>
                <th className="py-2 px-3">Processo</th>
                <th className="py-2 px-3">Comportamento ACF</th>
                <th className="py-2 px-3">Comportamento PACF</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-600">
              <tr>
                <td className="py-2 px-3 font-bold text-slate-900">AR(p)</td>
                <td className="py-2 px-3">Decaimento exponencial / senoidal</td>
                <td className="py-2 px-3">Corta abruptamente após o lag p</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-bold text-slate-900">MA(q)</td>
                <td className="py-2 px-3">Corta abruptamente após o lag q</td>
                <td className="py-2 px-3">Decaimento exponencial / senoidal</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-bold text-slate-900">ARMA(p,q)</td>
                <td className="py-2 px-3">Decai gradualmente</td>
                <td className="py-2 px-3">Decai gradualmente</td>
              </tr>
              <tr>
                <td className="py-2 px-3 font-bold text-slate-900">Não-Estacionária</td>
                <td className="py-2 px-3">Decaimento extremamente lento</td>
                <td className="py-2 px-3">Pico próximo de 1 no lag 1</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-600" />
            Árvore de Decisão Prática: Quando um Teste Falha
          </h3>
          <ul className="space-y-2 text-xs text-slate-700 leading-relaxed">
            <li>
              <strong>Variância crescente no nível (BP Temporal p &lt; α):</strong> aplique{" "}
              <code>Log Natural</code> ou <code>Box-Cox</code> antes de diferenciar.
            </li>
            <li>
              <strong>Raiz unitária (ADF/PP &amp; KPSS):</strong> aplique <code>1ª Diferença (Δyₜ)</code>; se houver força sazonal F_S &gt; 60%, teste <code>Δₛyₜ</code> ou <code>ΔΔₛyₜ</code>.
            </li>
            <li>
              <strong>Autocorrelação nos resíduos (Ljung-Box p &lt; α):</strong> aumente ordens <code>p, q</code> ou ative termos sazonais <code>(P, D, Q, s)</code>.
            </li>
            <li>
              <strong>Efeito ARCH significativo (ARCH-LM p &lt; α):</strong> a média condicional pode estar correta, mas a volatilidade exige um modelo <strong>GARCH(1,1)</strong> ou EGARCH.
            </li>
            <li>
              <strong>Resíduos não-normais (AD/JB p &lt; α):</strong> investigue outliers na aba <em>Visualizações</em>; previsões pontuais permanecem válidas, mas use intervalos via <em>bootstrap</em> ou inovação t-Student.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
