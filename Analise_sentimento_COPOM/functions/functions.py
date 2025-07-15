# Imports das bibliotecas
import pysentiment2 as ps
import pandas as pd
import numpy as np
import json
import urllib.request
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from bcb import sgs
import plotnine as p9
import os
from datetime import datetime, timedelta
import time
import pickle
import requests
from urllib.error import URLError
import warnings
warnings.filterwarnings('ignore')
import urllib
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

# === 1. funções para baixar atas do COPOM ===

def baixar_atas_incremental(quantidade=100, arquivo_progresso="atas_progresso.pkl"):
    """
    Baixa atas do COPOM de forma incremental, salvando o progresso.
    
    Args:
        quantidade: Número de atas para baixar
        arquivo_progresso: Nome do arquivo para salvar o progresso
    """
    
    print(f"🚀 Iniciando download de {quantidade} atas...")
    
    # 1. Buscar metadados das atas
    try:
        url_api = f"https://www.bcb.gov.br/api/servico/sitebcb/copomminutes/ultimas?quantidade={quantidade}&filtro=Id%20ne%20%27235%27"
        with urllib.request.urlopen(url_api) as response:
            dados = json.load(response)["conteudo"]
        
        df_base = pd.DataFrame(dados).assign(
            Url=lambda x: "https://www.bcb.gov.br/" + x.Url
        )
        print(f"✅ Metadados obtidos: {len(df_base)} atas encontradas")
        
    except Exception as e:
        print(f"❌ Erro ao buscar metadados: {e}")
        return None
    
    # 2. Verificar se existe progresso anterior
    atas_processadas = []
    indice_inicio = 0
    
    if os.path.exists(arquivo_progresso):
        try:
            with open(arquivo_progresso, 'rb') as f:
                atas_processadas = pickle.load(f)
            indice_inicio = len(atas_processadas)
            print(f"📂 Progresso anterior encontrado: {indice_inicio} atas já processadas")
        except:
            print("⚠️  Erro ao carregar progresso anterior, iniciando do zero")
    
    # 3. Processar atas restantes
    total = len(df_base)
    for i in range(indice_inicio, total):
        try:
            row = df_base.iloc[i]
            print(f"\n📄 Processando ata {i+1}/{total}: {row.get('Titulo', 'Sem título')}")
            print(f"🔗 URL: {row['Url']}")
            
            inicio = time.time()
            
            # Carregar e extrair texto do PDF
            loader = PyPDFLoader(row['Url'])
            documentos = loader.load()
            conteudo = " ".join(doc.page_content for doc in documentos)
            
            # Criar dicionário com todos os dados
            ata_processada = row.to_dict()
            ata_processada['conteudo'] = conteudo
            ata_processada['processado_em'] = datetime.now().isoformat()
            ata_processada['tempo_processamento'] = round(time.time() - inicio, 2)
            
            atas_processadas.append(ata_processada)
            
            # Salvar progresso após cada sucesso
            with open(arquivo_progresso, 'wb') as f:
                pickle.dump(atas_processadas, f)
            
            tempo = round(time.time() - inicio, 2)
            print(f"✅ Ata processada em {tempo}s | Total: {len(atas_processadas)}")
            
            # Pequena pausa para não sobrecarregar o servidor
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Erro ao processar ata {i+1}: {str(e)}")
            print("⏭️  Continuando com a próxima...")
            continue
    
    # 4. Converter para DataFrame final
    if atas_processadas:
        df_final = pd.DataFrame(atas_processadas)
        print(f"\n🎉 Processamento concluído!")
        print(f"📊 Total processado: {len(df_final)} atas")
        print(f"💾 Progresso salvo em: {arquivo_progresso}")
        
        return df_final
    else:
        print("❌ Nenhuma ata foi processada com sucesso")
        return None

def carregar_progresso(arquivo_progresso="atas_progresso.pkl"):
    """Carrega o progresso salvo como DataFrame"""
    try:
        with open(arquivo_progresso, 'rb') as f:
            atas = pickle.load(f)
        df = pd.DataFrame(atas)
        print(f"📂 Carregadas {len(df)} atas do arquivo de progresso")
        return df
    except FileNotFoundError:
        print("❌ Arquivo de progresso não encontrado")
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar progresso: {e}")
        return None

def salvar_csv(df, nome_arquivo="atas_copom.csv"):
    """Salva o DataFrame em CSV"""
    try:
        df.to_csv(nome_arquivo, index=False, encoding='utf-8')
        print(f"💾 Dados salvos em: {nome_arquivo}")
    except Exception as e:
        print(f"❌ Erro ao salvar CSV: {e}")


# === 3. funções para baixar os dados da selic de forma recursiva ===
def baixar_historico_ano_a_ano(codigo_sgs, nome_arquivo):

    """
    Baixa o histórico de uma série do SGS, um ano por vez, para evitar erros de conexão.
    
    Args:
        codigo_sgs (int): O código da série no sistema SGS do BCB.
        nome_arquivo (str): O nome do arquivo .csv para salvar os dados.
    
    Returns:
        pd.DataFrame: DataFrame com os dados históricos da série
    """
    
    # Se o arquivo já existe, carrega e retorna
    if os.path.exists(nome_arquivo):
        print(f"✅ O arquivo '{nome_arquivo}' já existe. Carregando dados existentes...")
        df_existente = pd.read_csv(nome_arquivo, index_col=0, parse_dates=True)
        return df_existente

    print(f"Iniciando download do histórico para o código {codigo_sgs}.")
    print("O processo será feito ano a ano para maior estabilidade...")

    # Define o intervalo de anos para buscar (últimos 11 anos para garantir 10 anos completos)
    ano_atual = datetime.now().year
    ano_inicial = ano_atual - 13
    
    lista_de_dataframes_anuais = []
    
    try:
        # Loop para buscar cada ano individualmente
        for ano in range(ano_inicial, ano_atual + 1):
            
            print(f"  -> Buscando dados para o ano de {ano}...")
            
            # Define o primeiro e o último dia do ano
            data_inicio = f'{ano}-01-01'
            data_fim = f'{ano}-12-31'
            
            # Realiza a chamada à API para o ano específico
            df_ano = sgs.get({ 'serie': codigo_sgs }, start=data_inicio, end=data_fim)
            
            if not df_ano.empty:
                lista_de_dataframes_anuais.append(df_ano)
            
            # Pausa de 1 segundo: essencial para não sobrecarregar o servidor do BCB
            time.sleep(1)

        if not lista_de_dataframes_anuais:
            print("❌ Nenhum dado foi retornado pela API. Verifique o código da série e sua conexão.")
            return pd.DataFrame()  # Retorna DataFrame vazio

        # Consolida todos os dataframes anuais em um único
        print("\nConsolidando todos os dados anuais...")
        df_final = pd.concat(lista_de_dataframes_anuais)
        
        # Remove duplicados, caso haja alguma sobreposição
        df_final = df_final[~df_final.index.duplicated(keep='first')]

        # Salva o resultado em um arquivo CSV
        df_final.to_csv(nome_arquivo)
        
        print(f"🎉 Sucesso! Histórico completo salvo no arquivo '{nome_arquivo}'.")
        print(f"Total de {len(df_final)} registros baixados de {df_final.index.min().year} a {df_final.index.max().year}.")
        
        return df_final

    except Exception as e:
        print(f"\n❌ Ocorreu um erro durante o processo: {e}")
        print("Verifique sua conexão com a internet ou tente novamente mais tarde.")
        return pd.DataFrame()  # Retorna DataFrame vazio em caso de erro



# === 3. EVOLUÇÃO TEMPORAL DO SENTIMENTO ===


def plot_sentimento_temporal(df):
    """Gráfico de linha mostrando evolução do sentimento ao longo do tempo"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Gráfico 1: Score numérico com média móvel
    ax1.plot(df['DataReferencia'], df['sentimento'], 'o-', alpha=0.7, label='Score de Sentimento')
    
    # Adiciona média móvel de 6 meses
    df_sorted = df.sort_values('DataReferencia')
    rolling_mean = df_sorted['sentimento'].rolling(window=6, center=True).mean()
    ax1.plot(df_sorted['DataReferencia'], rolling_mean, 'r-', linewidth=3, label='Média Móvel (6 períodos)')
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Evolução do Sentimento das Atas do COPOM', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Score de Sentimento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Classificação por cores
    colors = {'Positivo': 'green', 'Negativo': 'red', 'Neutro': 'gray'}
    for classificacao in df['classificacao'].unique():
        mask = df['classificacao'] == classificacao
        ax2.scatter(df[mask]['DataReferencia'], df[mask]['sentimento'], 
                   c=colors[classificacao], label=classificacao, s=60, alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Sentimento por Classificação', fontsize=14)
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Score de Sentimento')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# === 4. DISTRIBUIÇÃO E ESTATÍSTICAS ===
def plot_distribuicao_sentimento(df):
    """Análise da distribuição dos scores de sentimento"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histograma dos scores
    ax1.hist(df['sentimento'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df['sentimento'].mean(), color='red', linestyle='--', 
                label=f'Média: {df["sentimento"].mean():.3f}')
    ax1.set_title('Distribuição dos Scores de Sentimento')
    ax1.set_xlabel('Score de Sentimento')
    ax1.set_ylabel('Frequência')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Boxplot por classificação
    sns.boxplot(data=df, x='classificacao', y='sentimento', ax=ax2)
    ax2.set_title('Distribuição por Classificação')
    ax2.set_ylabel('Score de Sentimento')
    
    # Gráfico de pizza - proporção das classificações
    classificacao_counts = df['classificacao'].value_counts()
    ax3.pie(classificacao_counts.values, labels=classificacao_counts.index, autopct='%1.1f%%',
            colors=['lightgreen', 'lightcoral', 'lightgray'])
    ax3.set_title('Proporção das Classificações de Sentimento')
    
    # Q-Q plot para normalidade
    from scipy import stats
    stats.probplot(df['sentimento'], dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot - Teste de Normalidade')
    
    plt.tight_layout()
    return fig

# === 5. ANÁLISE TEMPORAL AVANÇADA ===
def plot_analise_temporal_avancada(df):
    """Heatmaps e análises sazonais"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Preparar dados temporais
    df_temp = df.copy()
    df_temp['Ano'] = df_temp['DataReferencia'].dt.year
    df_temp['Mes'] = df_temp['DataReferencia'].dt.month
    df_temp['Trimestre'] = df_temp['DataReferencia'].dt.quarter
    
    # Heatmap Ano x Mês
    pivot_mes = df_temp.groupby(['Ano', 'Mes'])['sentimento'].mean().unstack()
    sns.heatmap(pivot_mes, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax1)
    ax1.set_title('Heatmap: Sentimento Médio por Ano e Mês')
    
    # Boxplot por trimestre
    sns.boxplot(data=df_temp, x='Trimestre', y='sentimento', ax=ax2)
    ax2.set_title('Sentimento por Trimestre')
    ax2.set_xlabel('Trimestre')
    
    # Evolução anual
    sentimento_anual = df_temp.groupby('Ano')['sentimento'].agg(['mean', 'std']).reset_index()
    ax3.errorbar(sentimento_anual['Ano'], sentimento_anual['mean'], 
                yerr=sentimento_anual['std'], marker='o', capsize=5)
    ax3.set_title('Sentimento Médio Anual (com Desvio Padrão)')
    ax3.set_xlabel('Ano')
    ax3.set_ylabel('Sentimento Médio')
    ax3.grid(True, alpha=0.3)
    
    # Volatilidade (desvio padrão móvel)
    df_sorted = df_temp.sort_values('DataReferencia')
    rolling_std = df_sorted['sentimento'].rolling(window=6).std()
    ax4.plot(df_sorted['DataReferencia'], rolling_std, 'purple', linewidth=2)
    ax4.set_title('Volatilidade do Sentimento (Desvio Padrão Móvel)')
    ax4.set_xlabel('Data')
    ax4.set_ylabel('Volatilidade')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# === 6. CORRELAÇÃO COM SELIC (se disponível) ===
def plot_correlacao_selic(df_sentimento, df_selic):
    """Análise de correlação entre sentimento e taxa Selic"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Merge dos dados por data
    df_merged = pd.merge_asof(
        df_sentimento.sort_values('DataReferencia'),
        df_selic.reset_index().sort_values('Date').rename(columns={'Date': 'DataReferencia'}),
        on='DataReferencia',
        direction='backward'
    )
    
    # Gráfico dual-axis: Sentimento vs Selic
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(df_merged['DataReferencia'], df_merged['sentimento'], 'b-', label='Sentimento')
    line2 = ax1_twin.plot(df_merged['DataReferencia'], df_merged['serie'], 'r-', label='Selic')
    
    ax1.set_ylabel('Score de Sentimento', color='b')
    ax1_twin.set_ylabel('Taxa Selic (%)', color='r')
    ax1.set_title('Evolução: Sentimento vs Taxa Selic')
    
    # Combinar legendas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Scatter plot: correlação direta
    ax2.scatter(df_merged['sentimento'], df_merged['serie'], alpha=0.6)
    
    # Calcular e mostrar correlação
    correlation = df_merged[['sentimento', 'serie']].corr().iloc[0, 1]
    ax2.set_title(f'Correlação Sentimento vs Selic\n(r = {correlation:.3f})')
    ax2.set_xlabel('Score de Sentimento')
    ax2.set_ylabel('Taxa Selic (%)')
    
    # Adicionar linha de tendência
    z = np.polyfit(df_merged['sentimento'].dropna(), df_merged['serie'].dropna(), 1)
    p = np.poly1d(z)
    ax2.plot(df_merged['sentimento'], p(df_merged['sentimento']), "r--", alpha=0.8)
    
    # Análise de defasagens (lags)
    lags = range(-12, 13)
    correlations = []
    for lag in lags:
        if lag == 0:
            corr = df_merged[['sentimento', 'serie']].corr().iloc[0, 1]
        elif lag > 0:
            corr = df_merged['sentimento'].corr(df_merged['serie'].shift(lag))
        else:
            corr = df_merged['sentimento'].shift(-lag).corr(df_merged['serie'])
        correlations.append(corr)
    
    ax3.bar(lags, correlations)
    ax3.set_title('Correlação com Diferentes Defasagens')
    ax3.set_xlabel('Defasagem (meses)')
    ax3.set_ylabel('Correlação')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Diferenças e variações
    df_merged['delta_sentimento'] = df_merged['sentimento'].diff()
    df_merged['delta_selic'] = df_merged['serie'].diff()
    
    ax4.scatter(df_merged['delta_sentimento'], df_merged['delta_selic'], alpha=0.6)
    delta_corr = df_merged[['delta_sentimento', 'delta_selic']].corr().iloc[0, 1]
    ax4.set_title(f'Correlação das Variações\n(r = {delta_corr:.3f})')
    ax4.set_xlabel('Δ Sentimento')
    ax4.set_ylabel('Δ Selic')
    
    plt.tight_layout()
    return fig

# === 7. EVENTOS EXTREMOS E OUTLIERS ===
def plot_eventos_extremos(df):
    """Identificação e análise de eventos extremos"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Definir outliers (usando IQR)
    Q1 = df['sentimento'].quantile(0.25)
    Q3 = df['sentimento'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = df[(df['sentimento'] < limite_inferior) | (df['sentimento'] > limite_superior)]
    
    # Timeline com eventos extremos destacados
    ax1.plot(df['DataReferencia'], df['sentimento'], 'o-', alpha=0.7)
    ax1.scatter(outliers['DataReferencia'], outliers['sentimento'], 
                color='red', s=100, marker='x', label=f'Outliers ({len(outliers)})')
    ax1.axhline(y=limite_superior, color='red', linestyle='--', alpha=0.5, label='Limites IQR')
    ax1.axhline(y=limite_inferior, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Identificação de Eventos Extremos')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top 10 mais positivos e negativos
    top_positivos = df.nlargest(5, 'sentimento')
    top_negativos = df.nsmallest(5, 'sentimento')
    
    # Gráfico de barras dos extremos
    extremos = pd.concat([top_negativos, top_positivos])
    colors = ['red' if x < 0 else 'green' for x in extremos['sentimento']]
    
    ax2.barh(range(len(extremos)), extremos['sentimento'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(extremos)))
    ax2.set_yticklabels([f"{date.strftime('%Y-%m')}" for date in extremos['DataReferencia']])
    ax2.set_title('Top 5 Mais Positivos e Negativos')
    ax2.set_xlabel('Score de Sentimento')
    
    # Análise de clustering temporal
    df_sorted = df.sort_values('DataReferencia')
    df_sorted['sentimento_lag1'] = df_sorted['sentimento'].shift(1)
    
    ax3.scatter(df_sorted['sentimento_lag1'], df_sorted['sentimento'], alpha=0.6)
    ax3.set_title('Autocorrelação: Sentimento(t) vs Sentimento(t-1)')
    ax3.set_xlabel('Sentimento Anterior')
    ax3.set_ylabel('Sentimento Atual')
    
    # Calcular autocorrelação
    autocorr = df_sorted['sentimento'].autocorr(lag=1)
    ax3.text(0.05, 0.95, f'Autocorrelação: {autocorr:.3f}', 
             transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Rolling statistics
    window = 6
    df_sorted['rolling_mean'] = df_sorted['sentimento'].rolling(window).mean()
    df_sorted['rolling_std'] = df_sorted['sentimento'].rolling(window).std()
    
    ax4.fill_between(df_sorted['DataReferencia'], 
                     df_sorted['rolling_mean'] - df_sorted['rolling_std'],
                     df_sorted['rolling_mean'] + df_sorted['rolling_std'],
                     alpha=0.3, label='±1 Desvio Padrão')
    ax4.plot(df_sorted['DataReferencia'], df_sorted['rolling_mean'], 'b-', label='Média Móvel')
    ax4.plot(df_sorted['DataReferencia'], df_sorted['sentimento'], 'o', alpha=0.5, markersize=3)
    ax4.set_title(f'Bandas de Confiança (janela = {window})')
    ax4.legend()
    
    plt.tight_layout()
    return fig


# === 6. CORRELAÇÃO COM SELIC (se disponível) ===
def plot_correlacao_selic(df_sentimento, df_selic, salvar_como=None):
    """
    Plota análise gráfica da correlação entre sentimento das atas do COPOM e a taxa Selic.

    Parâmetros:
        df_sentimento (pd.DataFrame): Deve conter ['DataReferencia', 'sentimento']
        df_selic (pd.DataFrame ou Series): DataFrame com índice datetime (ou coluna 'Date') e coluna 'serie'
        salvar_como (str): Caminho para salvar a figura (ex: "saida.png"). Se None, apenas exibe.

    Retorna:
        matplotlib.figure.Figure: Figura contendo os 4 subgráficos
    """

    # ---- COPIAR os dataframes para preservar os originais
    sentimento = df_sentimento.copy()
    selic = df_selic.copy()

    # Garantir formato de data padronizado (sem timezone)
    sentimento['DataReferencia'] = pd.to_datetime(sentimento['DataReferencia']).dt.tz_localize(None)

    # Adaptar formato do DataFrame da Selic
    if 'Date' in selic.columns:
        selic = selic.rename(columns={'Date': 'DataReferencia'})
        selic['DataReferencia'] = pd.to_datetime(selic['DataReferencia']).dt.tz_localize(None)
    else:
        selic = selic.reset_index().rename(columns={selic.index.name: 'DataReferencia'})
        selic['DataReferencia'] = pd.to_datetime(selic['DataReferencia']).dt.tz_localize(None)

    # ---- Merge temporal por aproximação (para análise)
    df_merged = pd.merge_asof(
        sentimento.sort_values('DataReferencia'),
        selic.sort_values('DataReferencia'),
        on='DataReferencia',
        direction='backward'
    )

    # ---- Criar figura
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfico 1: Série temporal com dois eixos
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(df_merged['DataReferencia'], df_merged['sentimento'], 'b-', label='Sentimento')
    line2 = ax1_twin.plot(df_merged['DataReferencia'], df_merged['serie'], 'r-', label='Selic')
    ax1.set_ylabel('Sentimento', color='b')
    ax1_twin.set_ylabel('Selic (%)', color='r')
    ax1.set_title('Evolução: Sentimento vs Taxa Selic')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Gráfico 2: Correlação direta
    ax2.scatter(df_merged['sentimento'], df_merged['serie'], alpha=0.6)
    corr = df_merged[['sentimento', 'serie']].corr().iloc[0, 1]
    ax2.set_title(f'Correlação Sentimento vs Selic\n(r = {corr:.3f})')
    ax2.set_xlabel('Sentimento')
    ax2.set_ylabel('Selic (%)')

    z = np.polyfit(df_merged['sentimento'].dropna(), df_merged['serie'].dropna(), 1)
    ax2.plot(df_merged['sentimento'], np.poly1d(z)(df_merged['sentimento']), "r--", alpha=0.8)

    # Gráfico 3: Correlação com defasagens
    lags = range(-12, 13)
    lag_corrs = []
    for lag in lags:
        if lag == 0:
            value = corr
        elif lag > 0:
            value = df_merged['sentimento'].corr(df_merged['serie'].shift(lag))
        else:
            value = df_merged['sentimento'].shift(-lag).corr(df_merged['serie'])
        lag_corrs.append(value)

    ax3.bar(lags, lag_corrs, color='gray')
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Correlação com Defasagens')
    ax3.set_xlabel('Defasagem (meses)')
    ax3.set_ylabel('Correlação')

    # Gráfico 4: Correlação das variações
    df_merged['delta_sentimento'] = df_merged['sentimento'].diff()
    df_merged['delta_selic'] = df_merged['serie'].diff()
    delta_corr = df_merged[['delta_sentimento', 'delta_selic']].corr().iloc[0, 1]
    ax4.scatter(df_merged['delta_sentimento'], df_merged['delta_selic'], alpha=0.6)
    ax4.set_title(f'Correlação das Variações\n(r = {delta_corr:.3f})')
    ax4.set_xlabel('Δ Sentimento')
    ax4.set_ylabel('Δ Selic')

    plt.tight_layout()

    # Salvar se solicitado
    if salvar_como:
        fig.savefig(salvar_como, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico salvo em: {salvar_como}")

    return fig


# === 8. PLOTAR SENTIMENTO COPOM ===

def plotar_sentimento_copom(sentimento, titulo="Evolução do Sentimento nas Atas do COPOM"):
    """
    Plota a evolução temporal do sentimento nas atas do COPOM com áreas coloridas e pontos.

    Retorna:
        matplotlib.figure.Figure: objeto da figura (para salvar ou customizar)
    """
    
    # Verificações iniciais
    if sentimento is None or sentimento.empty:
        print("Erro: DataFrame de sentimento está vazio ou é None")
        return None
    
    if 'DataReferencia' not in sentimento.columns or 'sentimento' not in sentimento.columns:
        print("Erro: Colunas 'DataReferencia' e/ou 'sentimento' não encontradas")
        print(f"Colunas disponíveis: {sentimento.columns.tolist()}")
        return None
    
    try:
        # Ordenar os dados por data
        sentimento = sentimento.sort_values("DataReferencia").copy()
        
        # Remover valores nulos
        sentimento = sentimento.dropna(subset=['DataReferencia', 'sentimento'])
        
        if sentimento.empty:
            print("Erro: Não há dados válidos após remoção de valores nulos")
            return None

        # Criar figura e eixo
        fig, ax = plt.subplots(figsize=(16, 6))

        # Área verde para sentimentos positivos
        ax.fill_between(
            sentimento["DataReferencia"],
            0,
            sentimento["sentimento"],
            where=sentimento["sentimento"] > 0,
            interpolate=True,
            color="green",
            alpha=0.3,
            label="Sentimento Positivo"
        )

        # Área vermelha para sentimentos negativos
        ax.fill_between(
            sentimento["DataReferencia"],
            0,
            sentimento["sentimento"],
            where=sentimento["sentimento"] < 0,
            interpolate=True,
            color="red",
            alpha=0.3,
            label="Sentimento Negativo"
        )

        # Linha de sentimento
        ax.plot(
            sentimento["DataReferencia"],
            sentimento["sentimento"],
            color="black",
            linewidth=2,
            label="Evolução do Sentimento"
        )

        # Pontos
        ax.scatter(
            sentimento["DataReferencia"],
            sentimento["sentimento"],
            color="blue",
            s=50,
            alpha=0.7,
            zorder=5
        )

        # Linha horizontal no zero
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)

        # Títulos e formatação
        ax.set_title(titulo, fontsize=16, weight="bold", loc="center", pad=20)
        fig.suptitle("Análise temporal usando dicionário Loughran-McDonald", fontsize=12, y=0.92)
        ax.set_xlabel("Data da Reunião", fontsize=12)
        ax.set_ylabel("Índice de Sentimento", fontsize=12)
        
        # Rotação dos labels do eixo x
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.grid(axis='x', linestyle=':', alpha=0.3)

        # Legenda
        ax.legend(loc='upper right', fontsize=10)

        # Créditos
        fig.text(0.99, 0.01, "Dados: BCB", ha='right', fontsize=10)

        # Layout ajustado
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])

        print(f"Gráfico criado com sucesso! Período: {sentimento['DataReferencia'].min()} a {sentimento['DataReferencia'].max()}")
        print(f"Número de observações: {len(sentimento)}")
        
        return fig
        
    except Exception as e:
        print(f"Erro ao criar o gráfico: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# === 9. PLOTAR SENTIMENTO COPOM COM PRESIDENTES ===


def plotar_sentimento_copom_pres(sentimento, titulo="Evolução do Sentimento nas Atas do COPOM"):
    """
    Plota a evolução temporal do sentimento nas atas do COPOM com áreas coloridas, pontos
    e os períodos dos presidentes do Banco Central do Brasil de forma mais limpa.
    
    Args:
        sentimento: DataFrame com colunas 'DataReferencia' e 'sentimento'
        titulo: Título do gráfico
    
    Retorna:
        matplotlib.figure.Figure: objeto da figura (para salvar ou customizar)
    """
    
    # Verificações iniciais
    if sentimento is None or sentimento.empty:
        print("Erro: DataFrame de sentimento está vazio ou é None")
        return None
    
    if 'DataReferencia' not in sentimento.columns or 'sentimento' not in sentimento.columns:
        print("Erro: Colunas 'DataReferencia' e/ou 'sentimento' não encontradas")
        print(f"Colunas disponíveis: {sentimento.columns.tolist()}")
        return None
    
    try:
        # Ordenar os dados por data
        sentimento = sentimento.sort_values("DataReferencia").copy()
        
        # Remover valores nulos
        sentimento = sentimento.dropna(subset=['DataReferencia', 'sentimento'])
        
        # Garantir que as datas sejam compatíveis para comparação
        if hasattr(sentimento['DataReferencia'].dtype, 'tz') and sentimento['DataReferencia'].dtype.tz is not None:
            timezone = sentimento['DataReferencia'].dtype.tz
            cutoff_date = pd.Timestamp('2012-01-01').tz_localize(timezone)
        else:
            cutoff_date = pd.Timestamp('2012-01-01')
        
        # Filtrar dados a partir de 2012
        sentimento = sentimento[sentimento['DataReferencia'] >= cutoff_date]
        
        if sentimento.empty:
            print("Erro: Não há dados válidos após remoção de valores nulos e filtro para 2012+")
            return None
        
        # Definir períodos dos presidentes do BCB (apenas transições)
        if hasattr(sentimento['DataReferencia'].dtype, 'tz') and sentimento['DataReferencia'].dtype.tz is not None:
            timezone = sentimento['DataReferencia'].dtype.tz
            transicoes_presidentes = [
                {
                    'data': pd.Timestamp('2016-06-09').tz_localize(timezone),
                    'saida': 'Alexandre Tombini',
                    'entrada': 'Ilan Goldfajn'
                },
                {
                    'data': pd.Timestamp('2019-02-28').tz_localize(timezone),
                    'saida': 'Ilan Goldfajn',
                    'entrada': 'Roberto Campos Neto'
                },
                {
                    'data': pd.Timestamp('2025-01-01').tz_localize(timezone),
                    'saida': 'Roberto Campos Neto',
                    'entrada': 'Gabriel Galípolo'
                }
            ]
        else:
            transicoes_presidentes = [
                {
                    'data': pd.Timestamp('2016-06-09'),
                    'saida': 'Alexandre Tombini',
                    'entrada': 'Ilan Goldfajn'
                },
                {
                    'data': pd.Timestamp('2019-02-28'),
                    'saida': 'Ilan Goldfajn',
                    'entrada': 'Roberto Campos Neto'
                },
                {
                    'data': pd.Timestamp('2025-01-01'),
                    'saida': 'Roberto Campos Neto',
                    'entrada': 'Gabriel Galípolo'
                }
            ]
        
        # Criar figura e eixo
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Área verde para sentimentos positivos
        ax.fill_between(
            sentimento["DataReferencia"],
            0,
            sentimento["sentimento"],
            where=sentimento["sentimento"] > 0,
            interpolate=True,
            color="darkgreen",
            alpha=0.3,
            label="Sentimento Positivo"
        )
        
        # Área vermelha para sentimentos negativos
        ax.fill_between(
            sentimento["DataReferencia"],
            0,
            sentimento["sentimento"],
            where=sentimento["sentimento"] < 0,
            interpolate=True,
            color="darkred",
            alpha=0.3,
            label="Sentimento Negativo"
        )
        
        # Linha de sentimento
        ax.plot(
            sentimento["DataReferencia"],
            sentimento["sentimento"],
            color="black",
            linewidth=2.5,
            label="Evolução do Sentimento",
            zorder=10
        )
        
        # Pontos
        ax.scatter(
            sentimento["DataReferencia"],
            sentimento["sentimento"],
            color="navy",
            s=40,
            alpha=0.8,
            zorder=15,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Linha horizontal no zero
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
        
        # Adicionar linhas verticais e anotações para as transições
        data_min = sentimento["DataReferencia"].min()
        data_max = sentimento["DataReferencia"].max()
        
        # Determinar presidente atual no início do período
        presidente_inicial = "Alexandre Tombini"
        for transicao in transicoes_presidentes:
            if transicao['data'] <= data_min:
                presidente_inicial = transicao['entrada']
        
        # Adicionar anotação do presidente inicial na parte inferior
        y_max = sentimento['sentimento'].max()
        y_min = sentimento['sentimento'].min()
        range_y = y_max - y_min
        
        ax.text(data_min + pd.Timedelta(days=100), y_min - range_y * 0.1, 
               f" {presidente_inicial}", 
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.5),
               ha='left', va='top')
        
        # Adicionar linhas verticais para transições dentro do período
        for i, transicao in enumerate(transicoes_presidentes):
            if data_min <= transicao['data'] <= data_max:
                # Linha vertical
                ax.axvline(x=transicao['data'], 
                          color='red', 
                          linestyle='--', 
                          alpha=0.8, 
                          linewidth=1,
                          zorder=5)
                
                # Anotação da mudança - posicionada na parte inferior, colada ao eixo x
                y_pos = y_min - range_y * (0.15 + i * 0.12)  # Posição abaixo do gráfico
                ax.text(transicao['data'], y_pos,
                       f"{transicao['entrada']}\n{transicao['data'].strftime('%m/%Y')}",
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                       ha='left', va='top')
        
        # Ajustar limites do eixo Y para acomodar as anotações na parte inferior
        num_transicoes_no_periodo = len([t for t in transicoes_presidentes if data_min <= t['data'] <= data_max])
        if num_transicoes_no_periodo > 0:
            ax.set_ylim(y_min - range_y * (0.3 + num_transicoes_no_periodo * 0.12), y_max + range_y * 0.05)
        
        # Títulos e formatação
        ax.set_title(titulo, fontsize=18, weight="bold", loc="center", pad=25)
        fig.suptitle("Análise Temporal (2012+) | Mudanças de Presidência do BCB", 
                    fontsize=13, y=0.96, style='italic')
        ax.set_xlabel("Data da Reunião", fontsize=13, weight='bold')
        ax.set_ylabel("Índice de Sentimento", fontsize=13, weight='bold')
        
        # Formatação do eixo x
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        
        # Rotação dos labels do eixo x
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Grid melhorado
        ax.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.8)
        ax.grid(axis='x', linestyle=':', alpha=0.4, linewidth=0.6)
        
        # Legenda única e limpa
        legend_elements = [
            plt.Line2D([0], [0], color='darkgreen', alpha=0.3, linewidth=10, label='Sentimento Positivo'),
            plt.Line2D([0], [0], color='darkred', alpha=0.3, linewidth=10, label='Sentimento Negativo'),
            plt.Line2D([0], [0], color='black', linewidth=2.5, label='Evolução do Sentimento'),
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, label='Mudança de Presidente BCB')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                 title="Legenda", title_fontsize=12, framealpha=0.9)
        
        # Melhorar layout das bordas
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        
        # Créditos
        fig.text(0.99, 0.01, "Dados: BCB | Transições de Presidência BCB destacadas", 
                ha='right', fontsize=10, style='italic')
        
        # Layout ajustado - mais espaço na parte inferior para as anotações
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        
        print(f"Gráfico criado com sucesso! Período: {sentimento['DataReferencia'].min()} a {sentimento['DataReferencia'].max()}")
        print(f"Número de observações: {len(sentimento)}")
        print(f"Transições de presidência no período: {len([t for t in transicoes_presidentes if data_min <= t['data'] <= data_max])}")
        
        return fig
        
    except Exception as e:
        print(f"Erro ao criar o gráfico: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


#_________________________________________________________________________
# Função auxiliar para obter apenas os valores dos sentimentos em um df
#________________________________________________________________________
def preparar_dados_sentimento(df_sentimento: pd.DataFrame) -> pd.DataFrame:
    sentimento = df_sentimento.copy()

    sentimento['DataReferencia'] = pd.to_datetime(sentimento['DataReferencia']).dt.tz_localize(None)

    # Não há mais df_selic para fazer merge_asof, então df_initial_sentiment é o próprio sentimento
    df_initial_sentiment = sentimento.sort_values('DataReferencia').dropna(subset=['sentimento'])

    df_processed = df_initial_sentiment[['DataReferencia', 'sentimento']].copy()
    df_processed['Data_Mensal'] = df_processed['DataReferencia'].dt.to_period('M').dt.start_time
    df_processed = df_processed.rename(columns={'Data_Mensal': 'Data'})

    df_agregado_mensal = df_processed.groupby('Data').agg(
        sentimento=('sentimento', 'mean')
    ).reset_index()

    df_agregado_mensal = df_agregado_mensal.set_index('Data')

    df_com_meses_completos = df_agregado_mensal.resample('MS').first()
    df_com_meses_completos['sentimento'] = df_com_meses_completos['sentimento'].ffill()

    df_com_meses_completos = df_com_meses_completos.reset_index()
    # A coluna 'serie' não existe mais, então a linha para removê-la foi removida

    return df_com_meses_completos


#_________________________________________________________________________
# Função para plotar a correlação do sentimento com ipca
#________________________________________________________________________
def plot_correlacao_sentimento_ipca(df_merged_data, salvar_como=None):
    """
    Plota análise gráfica da correlação entre sentimento e IPCA.

    Parâmetros:
        df_merged_data (pd.DataFrame): Deve conter ['Data', 'sentimento', 'IPCA'].
                                       A coluna 'Data' deve ser do tipo datetime.
        salvar_como (str): Caminho para salvar a figura (ex: "saida.png"). Se None, apenas exibe.

    Retorna:
        matplotlib.figure.Figure: Figura contendo os 4 subgráficos.
    """

    # --- COPIAR o dataframe para preservar o original
    df = df_merged_data.copy()

    # Garantir formato de data padronizado (sem timezone, se houver)
    # Apenas se 'Data' não for o índice ou precisar de limpeza de timezone
    if 'Data' in df.columns:
        df['Data'] = pd.to_datetime(df['Data']).dt.tz_localize(None)

    # Definir 'Data' como índice para facilitar manipulações temporais
    df = df.set_index('Data')

    # --- Criar figura com 4 subgráficos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfico 1: Série temporal com dois eixos (Sentimento vs IPCA)
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(df.index, df['sentimento'], 'b-', label='Sentimento')
    line2 = ax1_twin.plot(df.index, df['IPCA'], 'r-', label='IPCA')
    ax1.set_ylabel('Sentimento', color='b')
    ax1_twin.set_ylabel('IPCA (%)', color='r')
    ax1.set_title('Evolução: Sentimento vs IPCA')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Gráfico 2: Correlação direta (Sentimento vs IPCA)
    ax2.scatter(df['sentimento'], df['IPCA'], alpha=0.6)
    corr = df[['sentimento', 'IPCA']].corr().iloc[0, 1]
    ax2.set_title(f'Correlação Sentimento vs IPCA\n(r = {corr:.3f})')
    ax2.set_xlabel('Sentimento')
    ax2.set_ylabel('IPCA (%)')

    # Adicionar linha de regressão linear
    # Usar .dropna() para garantir que não haja NaNs nos dados para polyfit
    sentimento_clean = df['sentimento'].dropna()
    ipca_clean = df['IPCA'].dropna()
    if not sentimento_clean.empty and not ipca_clean.empty:
        # Alinhar os índices para garantir que os dados correspondam
        common_index = sentimento_clean.index.intersection(ipca_clean.index)
        if not common_index.empty:
            z = np.polyfit(sentimento_clean.loc[common_index], ipca_clean.loc[common_index], 1)
            ax2.plot(sentimento_clean.loc[common_index], np.poly1d(z)(sentimento_clean.loc[common_index]), "r--", alpha=0.8)


    # Gráfico 3: Correlação com defasagens (Sentimento vs IPCA)
    lags = range(-12, 13) # Correlação com defasagens de -12 a +12 meses
    lag_corrs = []
    for lag in lags:
        # Calcula a correlação entre Sentimento e IPCA defasado
        # lag > 0: Sentimento de hoje vs IPCA de 'lag' meses atrás
        # lag < 0: Sentimento de 'lag' meses atrás vs IPCA de hoje
        if lag >= 0:
            value = df['sentimento'].corr(df['IPCA'].shift(lag))
        else: # lag < 0
            value = df['sentimento'].shift(-lag).corr(df['IPCA'])
        lag_corrs.append(value)

    ax3.bar(lags, lag_corrs, color='gray')
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Correlação com Defasagens')
    ax3.set_xlabel('Defasagem (meses)')
    ax3.set_ylabel('Correlação')

    # Gráfico 4: Correlação das variações (Δ Sentimento vs Δ IPCA)
    df['delta_sentimento'] = df['sentimento'].diff()
    df['delta_ipca'] = df['IPCA'].diff()
    delta_corr = df[['delta_sentimento', 'delta_ipca']].corr().iloc[0, 1]
    ax4.scatter(df['delta_sentimento'], df['delta_ipca'], alpha=0.6)
    ax4.set_title(f'Correlação das Variações\n(r = {delta_corr:.3f})')
    ax4.set_xlabel('Δ Sentimento')
    ax4.set_ylabel('Δ IPCA')

    plt.tight_layout() # Ajusta o layout para evitar sobreposição

    # Salvar se solicitado
    if salvar_como:
        fig.savefig(salvar_como, dpi=300, bbox_inches='tight')
        print(f"💾 Gráfico salvo em: {salvar_como}")
    else:
        plt.show() # Exibe o gráfico se não for para salvar

    return fig