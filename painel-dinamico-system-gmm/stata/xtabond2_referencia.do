/* ============================================================================
   Referência Stata (xtabond2) — Painel Dinâmico Two-Step System-GMM
   Baseado no artigo: "Economic Growth Channels From Human Capital:
   A Dynamic Panel Analysis for Brazil" (Revista Brasileira de Economia / SciELO)
   Autor: Silvio da Rosa Paula
   ============================================================================ */

* 1. Instalação de pacotes (executar apenas uma vez)
* ssc install xtabond2
* net install st0085_2.pkg
* ssc install asdoc

* 2. Configurações e carga da base (caminho relativo a partir de stata/)
mata: mata set matafavor speed, perm
set more off, perm

use "../data/BASE_CH.dta", clear
xtset id Ano

/* ----------------------------------------------------------------------------
   TABELA 3 — Two-Step System-GMM com PCA (controle de proliferação)
   ---------------------------------------------------------------------------- */
* Taxa de crescimento do Capital Humano Total
xtabond2 DlnY_L L.lnY_L Tx_K_L Tx_HC ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L Tx_HC) iv(ln_Gec popgr D_Crisis Trend n_prof_ah) ///
    twostep robust pca components(21)
estimates store reg1

* Log natural do Capital Humano Total
xtabond2 DlnY_L L.lnY_L Tx_K_L ln_HC ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L) iv(ln_Gec popgr D_Crisis Trend ln_HC) ///
    twostep robust pca components(21)
estimates store reg2

* Log natural + Taxa de crescimento (Total)
xtabond2 DlnY_L L.lnY_L Tx_K_L ln_HC Tx_HC ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L Tx_HC) iv(ln_Gec popgr D_Crisis Trend ln_HC n_prof_ah) ///
    twostep robust pca components(20)
estimates store reg3

* Taxa de crescimento — Educação Básica
xtabond2 DlnY_L L.lnY_L Tx_K_L Tx_HC_bh ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L Tx_HC_bh) iv(ln_Gec popgr D_Crisis Trend n_prof_bh) ///
    twostep robust pca components(21)
estimates store reg4

* Log natural — Educação Superior (Avançada)
xtabond2 DlnY_L L.lnY_L Tx_K_L ln_HC_ah ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L) iv(ln_Gec popgr D_Crisis Trend ln_HC_ah) ///
    twostep robust pca components(21)
estimates store reg5

* Log Educação Superior & Taxa Educação Básica
xtabond2 DlnY_L L.lnY_L Tx_K_L ln_HC_ah Tx_HC_bh ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L Tx_HC_bh) iv(ln_Gec popgr D_Crisis Trend ln_HC_ah n_prof_bh) ///
    twostep robust pca components(20)
estimates store reg6

* Taxa de crescimento — Educação Superior
xtabond2 DlnY_L L.lnY_L Tx_K_L Tx_HC_ah ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L Tx_HC_ah) iv(ln_Gec popgr D_Crisis Trend n_prof_ah) ///
    twostep robust pca components(21)
estimates store reg7

* Log natural — Educação Básica
xtabond2 DlnY_L L.lnY_L Tx_K_L ln_HC_bh ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L) iv(ln_Gec popgr D_Crisis Trend ln_HC_bh) ///
    twostep robust pca components(21)
estimates store reg8

* Log Educação Básica & Taxa Educação Superior
xtabond2 DlnY_L L.lnY_L Tx_K_L ln_HC_bh Tx_HC_ah ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L Tx_HC_ah) iv(ln_Gec popgr D_Crisis Trend ln_HC_bh n_prof_ah) ///
    twostep robust pca components(20)
estimates store reg9

/* ----------------------------------------------------------------------------
   TABELA 4 — Efeitos de Longo Prazo: beta / (1 - gamma) via Método Delta (nlcom)
   ---------------------------------------------------------------------------- */
xtabond2 lnY_L L.lnY_L Tx_K_L ln_HC ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L) iv(ln_Gec popgr D_Crisis Trend ln_HC) ///
    twostep robust pca components(21)
nlcom _b[ln_HC] / (1 - _b[L.lnY_L])

xtabond2 lnY_L L.lnY_L Tx_K_L ln_HC_ah ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L) iv(ln_Gec popgr D_Crisis Trend ln_HC_ah) ///
    twostep robust pca components(21)
nlcom _b[ln_HC_ah] / (1 - _b[L.lnY_L])

/* ----------------------------------------------------------------------------
   ROBUSTEZ (TABELAS 5–7) — Laglimits + Collapse (equivalente ao notebook Python)
   ---------------------------------------------------------------------------- */
* Benchmark direto comparado com pydynpd no notebook Python:
xtabond2 lnY_L L.lnY_L Tx_K_L Tx_HC ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L Tx_HC, lag(2 2) collapse) ///
    iv(ln_Gec popgr D_Crisis Trend n_prof_ah) twostep robust

xtabond2 lnY_L L.lnY_L Tx_K_L ln_HC ln_Gec popgr D_Crisis Trend, ///
    gmm(L.lnY_L Tx_K_L ln_HC, lag(2 7) collapse) ///
    iv(ln_Gec popgr D_Crisis Trend) twostep robust
