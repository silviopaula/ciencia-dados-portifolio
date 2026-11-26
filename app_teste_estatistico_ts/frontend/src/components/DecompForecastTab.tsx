import React from "react";
import { Compass, LineChart as LineIcon, TrendingUp } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnalysisPayload } from "../types";

interface DecompForecastTabProps {
  data: AnalysisPayload;
}

export const DecompForecastTab: React.FC<DecompForecastTabProps> = ({ data }) => {
  const { series_table, stl, arima, meta } = data;
  const bt = arima.backtest;

  // Combina últimos 60 pontos históricos + horizonte previsto para o gráfico de previsão
  const histSlice = series_table.slice(-60).map((r) => ({
    date: r.date,
    actual: r.value,
    forecast: null as number | null,
    band95: null as [number, number] | null,
    band80: null as [number, number] | null,
  }));

  const lastHist = series_table[series_table.length - 1];
  const fcSlice = arima.forecast.map((f) => ({
    date: f.date,
    actual: null as number | null,
    forecast: f.mean,
    band95:
      f.lower_95 !== null && f.upper_95 !== null ? ([f.lower_95, f.upper_95] as [number, number]) : null,
    band80:
      f.lower_80 !== null && f.upper_80 !== null ? ([f.lower_80, f.upper_80] as [number, number]) : null,
  }));

  const bridgePoint = lastHist
    ? [
        {
          date: lastHist.date,
          actual: lastHist.value,
          forecast: lastHist.value,
          band95: lastHist.value !== null ? ([lastHist.value, lastHist.value] as [number, number]) : null,
          band80: lastHist.value !== null ? ([lastHist.value, lastHist.value] as [number, number]) : null,
        },
      ]
    : [];

  const forecastChartData = [...histSlice.slice(0, -1), ...bridgePoint, ...fcSlice];

  return (
    <div className="space-y-6">
      {/* KPIs STL & Modelo SARIMA */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">Força Tendência (F_T)</div>
          <div className="text-2xl font-extrabold text-indigo-700 font-mono mt-1">
            {(stl.trend_strength * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5">Wang, Smith & Hyndman (2006)</p>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">Força Sazonalidade (F_S)</div>
          <div className="text-2xl font-extrabold text-emerald-700 font-mono mt-1">
            {(stl.seasonal_strength * 100).toFixed(1)}%
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5">Período s = {meta.seasonal_period}</p>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">Modelo Ajustado</div>
          <div className="text-sm font-extrabold text-slate-900 font-mono mt-1.5 break-all">
            {arima.spec_label}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            AIC: {arima.aic ?? "—"} • BIC: {arima.bic ?? "—"}
          </p>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">
            Backtest RMSE ({bt.horizon} per.)
          </div>
          <div className="text-2xl font-extrabold text-slate-900 font-mono mt-1">
            {bt.rmse ?? "—"}
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5">Seasonal Naive: {bt.naive_rmse ?? "—"}</p>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">Ganho vs Naive</div>
          <div
            className={`text-2xl font-extrabold font-mono mt-1 ${
              (bt.skill_vs_naive_pct ?? 0) >= 0 ? "text-emerald-600" : "text-rose-600"
            }`}
          >
            {bt.skill_vs_naive_pct !== null
              ? `${bt.skill_vs_naive_pct >= 0 ? "+" : ""}${bt.skill_vs_naive_pct.toFixed(1)}%`
              : "—"}
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5">
            MAPE: {bt.mape !== null ? `${bt.mape.toFixed(2)}%` : "N/A"}
          </p>
        </div>
      </div>

      {/* Decomposição STL Robusta (4 painéis) */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Compass className="w-4 h-4 text-indigo-600" />
            Decomposição STL Robusta ({meta.transform_label})
          </h3>
          <p className="text-xs text-slate-500">
            Separação não-paramétrica via LOESS robusto em Tendência, Sazonalidade (s = {meta.seasonal_period}) e Resíduo Irregular.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[
            { key: "value", title: "1. Série Analisada", color: "#2563eb" },
            { key: "stl_trend", title: "2. Componente de Tendência", color: "#4f46e5" },
            { key: "stl_seasonal", title: "3. Componente Sazonal", color: "#059669" },
            { key: "stl_remainder", title: "4. Resíduo Irregular (STL)", color: "#e11d48" },
          ].map((panel) => (
            <div key={panel.key} className="border border-slate-100 rounded-lg p-3 bg-slate-50/40">
              <h4 className="text-xs font-bold text-slate-700 mb-1.5">{panel.title}</h4>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series_table} syncId="stlSync">
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={35} />
                    <YAxis tick={{ fontSize: 10 }} width={52} />
                    <Tooltip />
                    {panel.key === "stl_remainder" || panel.key === "stl_seasonal" ? (
                      <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                    ) : null}
                    <Line
                      type="monotone"
                      dataKey={panel.key}
                      stroke={panel.color}
                      strokeWidth={1.9}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Previsão SARIMA com IC 80% / IC 95% */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
        <div className="mb-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-rose-600" />
            Projeção {arima.spec_label} ({arima.forecast.length} períodos) com Bandas IC 80% e IC 95%
          </h3>
          <p className="text-xs text-slate-500">
            Exibindo os últimos 60 períodos observados conectados à previsão pontual e intervalos probabilísticos.
          </p>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={forecastChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={25} />
              <YAxis tick={{ fontSize: 11 }} width={58} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area
                type="monotone"
                dataKey="band95"
                name="IC 95%"
                fill="#fecdd3"
                stroke="none"
                fillOpacity={0.55}
              />
              <Area
                type="monotone"
                dataKey="band80"
                name="IC 80%"
                fill="#fda4af"
                stroke="none"
                fillOpacity={0.55}
              />
              <Line
                type="monotone"
                dataKey="actual"
                name="Histórico"
                stroke="#2563eb"
                strokeWidth={2.2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="forecast"
                name="Previsão Pontual"
                stroke="#e11d48"
                strokeWidth={2.5}
                strokeDasharray="5 4"
                dot={{ r: 2.5 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Backtest Out-of-Sample vs Seasonal Naive */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
        <div className="mb-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <LineIcon className="w-4 h-4 text-emerald-600" />
            Validação Out-of-Sample (Últimos {bt.horizon} Períodos): {arima.spec_label} vs. Seasonal Naive
          </h3>
          <p className="text-xs text-slate-500">
            Modelo reestimado estritamente na amostra de treino (sem ver os últimos {bt.horizon} pontos) para auditar poder preditivo real.
          </p>
        </div>
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={bt.series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} width={58} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="actual" name="Real (OOS)" stroke="#0f172a" strokeWidth={2.5} />
              <Line type="monotone" dataKey="sarima" name={arima.spec_label} stroke="#2563eb" strokeWidth={2.2} />
              <Line
                type="monotone"
                dataKey="naive"
                name="Baseline Seasonal Naive"
                stroke="#94a3b8"
                strokeDasharray="4 4"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
