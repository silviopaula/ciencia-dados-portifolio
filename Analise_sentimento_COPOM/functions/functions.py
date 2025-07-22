#################################################################################################
# Imports das bibliotecas
#################################################################################################

import pysentiment2 as ps
import pandas as pd
import numpy as np
import json
import urllib
import urllib.request
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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sidrapy import get_table


#################################################################################################
# funções para baixar atas do COPOM
#################################################################################################

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
        print(f"Metadados obtidos: {len(df_base)} atas encontradas")
        
    except Exception as e:
        print(f"Erro ao buscar metadados: {e}")
        return None
    
    # 2. Verificar se existe progresso anterior
    atas_processadas = []
    indice_inicio = 0
    
    if os.path.exists(arquivo_progresso):
        try:
            with open(arquivo_progresso, 'rb') as f:
                atas_processadas = pickle.load(f)
            indice_inicio = len(atas_processadas)
            print(f"Progresso anterior encontrado: {indice_inicio} atas já processadas")
        except:
            print("Erro ao carregar progresso anterior, iniciando do zero")
    
    # 3. Processar atas restantes
    total = len(df_base)
    for i in range(indice_inicio, total):
        try:
            row = df_base.iloc[i]
            print(f"Processando ata {i+1}/{total}: {row.get('Titulo', 'Sem título')}")
            print(f"URL: {row['Url']}")
            
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
            print(f"Ata processada em {tempo}s | Total: {len(atas_processadas)}")
            
            # Pequena pausa para não sobrecarregar o servidor
            time.sleep(1)
            
        except Exception as e:
            print(f"Erro ao processar ata {i+1}: {str(e)}")
            print("Continuando com a próxima...")
            continue
    
    # 4. Converter para DataFrame final
    if atas_processadas:
        df_final = pd.DataFrame(atas_processadas)
        print(f"Processamento concluído!")
        print(f"Total processado: {len(df_final)} atas")
        print(f"Progresso salvo em: {arquivo_progresso}")
        
        return df_final
    else:
        print("Nenhuma ata foi processada com sucesso")
        return None

#################################################################################################
# Função para carregar atas .pkl
#################################################################################################

def carregar_progresso(arquivo_progresso="atas_progresso.pkl"):
    """Carrega o progresso salvo como DataFrame"""
    try:
        with open(arquivo_progresso, 'rb') as f:
            atas = pickle.load(f)
        df = pd.DataFrame(atas)
        print(f"Carregadas {len(df)} atas do arquivo de progresso")
        return df
    except FileNotFoundError:
        print("Arquivo de progresso não encontrado")
        return None
    except Exception as e:
        print(f"Erro ao carregar progresso: {e}")
        return None

def salvar_csv(df, nome_arquivo="atas_copom.csv"):
    """Salva o DataFrame em CSV"""
    try:
        df.to_csv(nome_arquivo, index=False, encoding='utf-8')
        print(f"Dados salvos em: {nome_arquivo}")
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")


#################################################################################################
# funções para baixar os dados da selic de forma recursiva
#################################################################################################

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
        print(f"O arquivo '{nome_arquivo}' já existe. Carregando dados existentes...")
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
            print("Nenhum dado foi retornado pela API. Verifique o código da série e sua conexão.")
            return pd.DataFrame()  # Retorna DataFrame vazio

        # Consolida todos os dataframes anuais em um único
        print("\nConsolidando todos os dados anuais...")
        df_final = pd.concat(lista_de_dataframes_anuais)
        
        # Remove duplicados, caso haja alguma sobreposição
        df_final = df_final[~df_final.index.duplicated(keep='first')]

        # Salva o resultado em um arquivo CSV
        df_final.to_csv(nome_arquivo)
        
        print(f"Sucesso! Histórico completo salvo no arquivo '{nome_arquivo}'.")
        print(f"Total de {len(df_final)} registros baixados de {df_final.index.min().year} a {df_final.index.max().year}.")
        
        return df_final

    except Exception as e:
        print(f"\nOcorreu um erro durante o processo: {e}")
        print("Verifique sua conexão com a internet ou tente novamente mais tarde.")
        return pd.DataFrame()  # Retorna DataFrame vazio em caso de erro


#################################################################################################
# Função para classificação do sentimento
#################################################################################################

def classificar_sentimento(df_sentimento):
    """
    Classificação robusta para análise de política monetária.
    Baseada em thresholds absolutos com validação estatística.
    """
    
    # Analisar a distribuição
    sentimentos = df_sentimento['sentimento']
    std = sentimentos.std()
    mean = sentimentos.mean()
    
    # Threshold adaptativo: maior entre 0.05 ou 0.5*std
    threshold_estatistico = 0.5 * std
    threshold_economico = 0.05
    threshold_final = max(threshold_estatistico, threshold_economico)
    
    print(f"📊 Análise dos thresholds:")
    print(f"   Desvio padrão: {std:.4f}")
    print(f"   0.5 × std: {threshold_estatistico:.4f}")
    print(f"   Threshold econômico: {threshold_economico:.4f}")
    print(f"   Threshold final: ±{threshold_final:.4f}")
    
    # Função de classificação
    def classificar_score(score):
        if score > threshold_final:
            return "Positivo"
        elif score < -threshold_final:
            return "Negativo"
        else:
            return "Neutro"
    
    # Verificar distribuição resultante
    classificacoes_temp = sentimentos.apply(classificar_score)
    distribuicao = classificacoes_temp.value_counts(normalize=True) * 100
    
    print(f"\n📈 Distribuição resultante:")
    for categoria, pct in distribuicao.items():
        print(f"   {categoria}: {pct:.1f}%")
    
    return classificar_score, threshold_final


#################################################################################################
# Função para plotar a evolução do sentimento
#################################################################################################

def plot_sentimento_temporal(df):
    """Gráfico de linha mostrando evolução do sentimento ao longo do tempo usando Plotly"""
    
    # Criar subplots com 2 linhas
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Evolução do Sentimento das Atas do COPOM', 'Sentimento por Classificação'),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Preparar dados ordenados para média móvel
    df_sorted = df.sort_values('DataReferencia')
    rolling_mean = df_sorted['sentimento'].rolling(window=6, center=True).mean()
    
    # Gráfico 1: Score numérico com média móvel
    # Linha principal do sentimento
    fig.add_trace(
        go.Scatter(
            x=df['DataReferencia'],
            y=df['sentimento'],
            mode='lines+markers',
            name='Score de Sentimento',
            line=dict(color='blue', width=2),
            marker=dict(size=6, opacity=0.7),
            hovertemplate='<b>Data:</b> %{x}<br><b>Sentimento:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Média móvel
    fig.add_trace(
        go.Scatter(
            x=df_sorted['DataReferencia'],
            y=rolling_mean,
            mode='lines',
            name='Média Móvel (6 períodos)',
            line=dict(color='red', width=3),
            hovertemplate='<b>Data:</b> %{x}<br><b>Média Móvel:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Linha horizontal em y=0 (primeiro gráfico)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)
    
    # Gráfico 2: Classificação por cores
    colors = {'Positivo': 'green', 'Negativo': 'red', 'Neutro': 'gray'}
    
    for classificacao in df['classificacao'].unique():
        mask = df['classificacao'] == classificacao
        df_filtered = df[mask]
        
        fig.add_trace(
            go.Scatter(
                x=df_filtered['DataReferencia'],
                y=df_filtered['sentimento'],
                mode='markers',
                name=f'{classificacao}',
                marker=dict(
                    color=colors[classificacao],
                    size=8,
                    opacity=0.8
                ),
                hovertemplate=f'<b>{classificacao}</b><br><b>Data:</b> %{{x}}<br><b>Sentimento:</b> %{{y:.3f}}<extra></extra>',
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Linha horizontal em y=0 (segundo gráfico)
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
    
    # Atualizar layout
    fig.update_layout(
        height=700,
        title={
            'text': 'Análise Temporal do Sentimento - COPOM',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Atualizar eixos
    fig.update_xaxes(
        title_text="Data",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Score de Sentimento",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Score de Sentimento",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )
    
    return fig


#################################################################################################
# Função para plotar a distribuição estatistica dos sentimento
#################################################################################################


def plot_distribuicao_sentimento(df):
    """Análise da distribuição dos scores de sentimento usando Plotly"""
    
    # Criar subplots com layout 2x2
    # Definir specs para incluir gráfico de pizza
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribuição dos Scores de Sentimento',
            'Distribuição por Classificação', 
            'Proporção das Classificações de Sentimento',
            'Q-Q Plot - Teste de Normalidade'
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "domain"}, {"type": "xy"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Histograma dos scores
    fig.add_trace(
        go.Histogram(
            x=df['sentimento'],
            nbinsx=30,
            name='Distribuição',
            marker=dict(
                color='skyblue',
                line=dict(color='black', width=1)
            ),
            opacity=0.7,
            hovertemplate='<b>Intervalo:</b> %{x}<br><b>Frequência:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Adicionar linha da média
    media_sentimento = df['sentimento'].mean()
    fig.add_vline(
        x=media_sentimento,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Média: {media_sentimento:.3f}",
        annotation_position="top",
        row=1, col=1
    )
    
    # 2. Boxplot por classificação
    cores_box = {'Positivo': 'lightgreen', 'Negativo': 'lightcoral', 'Neutro': 'lightgray'}
    
    for i, classificacao in enumerate(df['classificacao'].unique()):
        dados_classe = df[df['classificacao'] == classificacao]['sentimento']
        
        fig.add_trace(
            go.Box(
                y=dados_classe,
                name=classificacao,
                marker_color=cores_box.get(classificacao, 'lightblue'),
                boxpoints='outliers',
                hovertemplate=f'<b>{classificacao}</b><br><b>Valor:</b> %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # 3. Gráfico de pizza - proporção das classificações
    classificacao_counts = df['classificacao'].value_counts()
    cores_pizza = [cores_box.get(label, 'lightblue') for label in classificacao_counts.index]
    
    fig.add_trace(
        go.Pie(
            labels=classificacao_counts.index,
            values=classificacao_counts.values,
            marker=dict(colors=cores_pizza),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Quantidade: %{value}<br>Percentual: %{percent}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Q-Q Plot para normalidade
    # Calcular os quantis teóricos e observados
    sorted_data = np.sort(df['sentimento'])
    n = len(sorted_data)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
    
    # Calcular linha de referência
    slope, intercept, r_value, _, _ = stats.linregress(theoretical_quantiles, sorted_data)
    line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    line_y = slope * line_x + intercept
    
    # Pontos do Q-Q plot
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            name='Dados Observados',
            marker=dict(color='blue', size=6),
            hovertemplate='<b>Quantil Teórico:</b> %{x:.3f}<br><b>Quantil Observado:</b> %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Linha de referência
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            name=f'Linha de Referência (R²={r_value**2:.3f})',
            line=dict(color='red', dash='dash'),
            hovertemplate='<b>Linha de Referência</b><br>R²: %{text}<extra></extra>',
            text=[f'{r_value**2:.3f}'] * len(line_x)
        ),
        row=2, col=2
    )
    
    # Atualizar layout geral
    fig.update_layout(
        height=800,
        title={
            'text': 'Análise de Distribuição do Sentimento - COPOM',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        showlegend=True,
        template='plotly_white'
    )
    
    # Atualizar eixos específicos
    # Histograma
    fig.update_xaxes(title_text="Score de Sentimento", showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Frequência", showgrid=True, row=1, col=1)
    
    # Boxplot
    fig.update_xaxes(title_text="Classificação", showgrid=True, row=1, col=2)
    fig.update_yaxes(title_text="Score de Sentimento", showgrid=True, row=1, col=2)
    
    # Q-Q Plot
    fig.update_xaxes(title_text="Quantis Teóricos", showgrid=True, row=2, col=2)
    fig.update_yaxes(title_text="Quantis Observados", showgrid=True, row=2, col=2)
    
    return fig

#################################################################################################
# Função para calcular a distribuição estatistica dos sentimento
#################################################################################################

def estatisticas_sentimento(df):
    """Retorna estatísticas descritivas do sentimento como DataFrame"""
    
    # Calcular estatísticas
    stats_dict = {
        'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Assimetria', 'Curtose'],
        'Valor': [
            df['sentimento'].mean(),
            df['sentimento'].median(),
            df['sentimento'].std(),
            df['sentimento'].min(),
            df['sentimento'].max(),
            df['sentimento'].skew(),
            df['sentimento'].kurtosis()
        ],
        'Interpretação': [
            'Valor central dos sentimentos',
            'Valor que divide os dados ao meio',
            'Dispersão dos dados em torno da média',
            'Menor valor observado',
            'Maior valor observado',
            'Assimetria da distribuição (0 = simétrica)',
            'Concentração em torno da média (3 = normal)'
        ]
    }
    
    # Teste de normalidade Shapiro-Wilk
    if len(df) <= 5000:  # Shapiro-Wilk funciona bem para n <= 5000
        shapiro_stat, shapiro_p = stats.shapiro(df['sentimento'])
        
        # Adicionar às listas
        stats_dict['Estatística'].extend(['Shapiro-Wilk Statistic', 'Shapiro-Wilk p-value'])
        stats_dict['Valor'].extend([shapiro_stat, shapiro_p])
        stats_dict['Interpretação'].extend([
            'Estatística do teste de normalidade',
            'p-valor (< 0.05 = não normal)'
        ])
    
    # Criar DataFrame
    df_stats = pd.DataFrame(stats_dict)
    
    # Arredondar valores para melhor visualização
    df_stats['Valor'] = df_stats['Valor'].round(4)
    
    return df_stats

#################################################################################################
# Função para plotar a análise temporal avançada
#################################################################################################


def plot_analise_temporal_avancada(df):
    """Heatmaps e análises sazonais usando Plotly"""
    
    # Preparar dados temporais
    df_temp = df.copy()
    df_temp['Ano'] = df_temp['DataReferencia'].dt.year
    df_temp['Mes'] = df_temp['DataReferencia'].dt.month
    df_temp['Trimestre'] = df_temp['DataReferencia'].dt.quarter
    
    # Criar subplots com diferentes tipos
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Heatmap: Sentimento Médio por Ano e Mês',
            'Sentimento por Trimestre',
            'Sentimento Médio Anual (com Desvio Padrão)',
            'Volatilidade do Sentimento (Desvio Padrão Móvel)'
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # === 1. HEATMAP ANO X MÊS ===
    pivot_mes = df_temp.groupby(['Ano', 'Mes'])['sentimento'].mean().unstack()
    
    # Preencher valores ausentes com NaN para o heatmap
    anos = sorted(df_temp['Ano'].unique())
    meses = list(range(1, 13))
    
    # Criar matriz completa e texto das anotações
    z_matrix = []
    text_matrix = []
    y_labels = []
    
    for ano in anos:
        row_z = []
        row_text = []
        for mes in meses:
            if ano in pivot_mes.index and mes in pivot_mes.columns:
                valor = pivot_mes.loc[ano, mes]
                if pd.notna(valor):
                    row_z.append(valor)
                    row_text.append(f'{valor:.3f}')
                else:
                    row_z.append(None)
                    row_text.append('')
            else:
                row_z.append(None)
                row_text.append('')
        z_matrix.append(row_z)
        text_matrix.append(row_text)
        y_labels.append(str(ano))
    
    # Criar labels dos meses
    meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    
    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=meses_nomes,
            y=y_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 10, "color": "black"},
            colorscale='RdYlGn',
            zmid=0,
            showscale=True,
            colorbar=dict(
                title="Sentimento",
                x=0.48,
                len=0.4
            ),
            hovertemplate='<b>Ano:</b> %{y}<br><b>Mês:</b> %{x}<br><b>Sentimento:</b> %{z:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # === 2. BOXPLOT POR TRIMESTRE ===
    trimestres = sorted(df_temp['Trimestre'].unique())
    cores_trim = ['#4682B4', '#32CD32', '#FF6347', '#8A2BE2']  # SteelBlue, LimeGreen, Tomato, BlueViolet
    
    for i, trim in enumerate(trimestres):
        dados_trim = df_temp[df_temp['Trimestre'] == trim]['sentimento']
        
        fig.add_trace(
            go.Box(
                y=dados_trim,
                name=f'Q{trim}',
                marker_color=cores_trim[i] if i < len(cores_trim) else '#808080',
                boxpoints='outliers',
                hovertemplate=f'<b>Trimestre {trim}</b><br><b>Valor:</b> %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # === 3. EVOLUÇÃO ANUAL COM BARRAS DE ERRO ===
    sentimento_anual = df_temp.groupby('Ano')['sentimento'].agg(['mean', 'std']).reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=sentimento_anual['Ano'],
            y=sentimento_anual['mean'],
            error_y=dict(
                type='data',
                array=sentimento_anual['std'],
                visible=True,
                color='rgba(0,0,0,0.3)',
                thickness=2,
                width=5
            ),
            mode='markers+lines',
            marker=dict(size=8, color='steelblue'),
            line=dict(color='steelblue', width=2),
            name='Sentimento Anual',
            hovertemplate='<b>Ano:</b> %{x}<br><b>Média:</b> %{y:.3f}<br><b>Desvio:</b> %{error_y.array:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # === 4. VOLATILIDADE (DESVIO PADRÃO MÓVEL) ===
    df_sorted = df_temp.sort_values('DataReferencia')
    rolling_std = df_sorted['sentimento'].rolling(window=6).std()
    
    fig.add_trace(
        go.Scatter(
            x=df_sorted['DataReferencia'],
            y=rolling_std,
            mode='lines',
            line=dict(color='purple', width=2),
            name='Volatilidade',
            hovertemplate='<b>Data:</b> %{x}<br><b>Volatilidade:</b> %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # === ATUALIZAR LAYOUT ===
    fig.update_layout(
        height=800,
        title={
            'text': 'Análise Temporal Avançada do Sentimento - COPOM',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        showlegend=False,  # Remover legendas dos subplots individuais
        template='plotly_white'
    )
    
    # Atualizar eixos específicos
    # Heatmap
    fig.update_xaxes(title_text="Mês", row=1, col=1)
    fig.update_yaxes(title_text="Ano", row=1, col=1)
    
    # Boxplot
    fig.update_xaxes(title_text="Trimestre", row=1, col=2)
    fig.update_yaxes(title_text="Sentimento", row=1, col=2)
    
    # Evolução anual
    fig.update_xaxes(title_text="Ano", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Sentimento Médio", showgrid=True, row=2, col=1)
    
    # Volatilidade
    fig.update_xaxes(title_text="Data", showgrid=True, row=2, col=2)
    fig.update_yaxes(title_text="Volatilidade", showgrid=True, row=2, col=2)
    
    return fig

#################################################################################################
# função para analisar sazonalidade
#################################################################################################

# Função auxiliar para análise sazonal detalhada
def analise_sazonalidade(df):
    """Retorna DataFrame com análise sazonal detalhada"""
    
    df_temp = df.copy()
    df_temp['Ano'] = df_temp['DataReferencia'].dt.year
    df_temp['Mes'] = df_temp['DataReferencia'].dt.month
    df_temp['Trimestre'] = df_temp['DataReferencia'].dt.quarter
    df_temp['Semestre'] = df_temp['DataReferencia'].dt.month.apply(lambda x: 1 if x <= 6 else 2)
    
    # Análise por diferentes períodos
    analises = {}
    
    # Por mês
    analises['mensal'] = df_temp.groupby('Mes')['sentimento'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    
    # Por trimestre
    analises['trimestral'] = df_temp.groupby('Trimestre')['sentimento'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    
    # Por semestre
    analises['semestral'] = df_temp.groupby('Semestre')['sentimento'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    
    # Por ano
    analises['anual'] = df_temp.groupby('Ano')['sentimento'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    
    return analises

#################################################################################################
# Função para plotar eventos extremos e identificar outliers
#################################################################################################

def plot_eventos_extremos(df):
    """Identificação e análise de eventos extremos usando Plotly"""
    
    # Preparar dados dos extremos (usado em múltiplas seções)
    top_positivos = df.nlargest(5, 'sentimento')
    top_negativos = df.nsmallest(5, 'sentimento')
    extremos = pd.concat([top_negativos, top_positivos]).sort_values('sentimento')
    labels_datas = [f"{date.strftime('%Y-%m')}" for date in extremos['DataReferencia']]
    
    # Criar subplots 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Identificação de Eventos Extremos',
            'Top 5 Mais Positivos e Negativos - Lollipop',
            'Autocorrelação: Sentimento(t) vs Sentimento(t-1)',
            'Bandas de Confiança (janela = 6)'
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        row_heights=[0.5, 0.5],  # Altura igual para todas as linhas
        column_widths=[0.5, 0.5]  # Largura igual para todas as colunas
    )
    
    # === CALCULAR OUTLIERS (IQR) ===
    Q1 = df['sentimento'].quantile(0.25)
    Q3 = df['sentimento'].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = df[(df['sentimento'] < limite_inferior) | (df['sentimento'] > limite_superior)]
    dados_normais = df[(df['sentimento'] >= limite_inferior) & (df['sentimento'] <= limite_superior)]
    
    # === 1. TIMELINE COM EVENTOS EXTREMOS ===
    # Dados normais
    fig.add_trace(
        go.Scatter(
            x=dados_normais['DataReferencia'],
            y=dados_normais['sentimento'],
            mode='lines+markers',
            name='Sentimento Normal',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6, opacity=0.7),
            hovertemplate='<b>Data:</b> %{x}<br><b>Sentimento:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Outliers
    if len(outliers) > 0:
        fig.add_trace(
            go.Scatter(
                x=outliers['DataReferencia'],
                y=outliers['sentimento'],
                mode='markers',
                name=f'Outliers ({len(outliers)})',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='red',
                    line=dict(width=2)
                ),
                hovertemplate='<b>OUTLIER</b><br><b>Data:</b> %{x}<br><b>Sentimento:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Linhas dos limites IQR
    fig.add_hline(
        y=limite_superior,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text=f"Limite Superior: {limite_superior:.3f}",
        annotation_position="right",
        row=1, col=1
    )
    
    fig.add_hline(
        y=limite_inferior,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text=f"Limite Inferior: {limite_inferior:.3f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # === 2. GRÁFICO LOLLIPOP DOS EXTREMOS ===
    # Criar cores
    cores = ['#FF4444' if x < 0 else '#44AA44' for x in extremos['sentimento']]
    
    # Linhas do lollipop
    for i, (idx, row) in enumerate(extremos.iterrows()):
        fig.add_trace(go.Scatter(
            x=[0, row['sentimento']],
            y=[i, i],
            mode='lines',
            line=dict(color=cores[i], width=4),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=2)
    
    # Pontos do lollipop
    fig.add_trace(go.Scatter(
        x=extremos['sentimento'],
        y=list(range(len(extremos))),
        mode='markers+text',
        marker=dict(
            size=12,
            color=cores,
            line=dict(width=2, color='white')
        ),
        text=[f"{val:.3f}" for val in extremos['sentimento']],
        textposition='middle right',
        name='Extremos',
        hovertemplate='<b>Data:</b> %{customdata}<br><b>Sentimento:</b> %{x:.3f}<extra></extra>',
        customdata=labels_datas,
        textfont=dict(size=9, color='black'),
        showlegend=False
    ), row=1, col=2)
    
    # Linha vertical no zero para o lollipop
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, len(extremos)-1],
        mode='lines',
        line=dict(dash='dash', color='black', width=1),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=2)
    
    # === 3. AUTOCORRELAÇÃO ===
    df_sorted = df.sort_values('DataReferencia').copy()
    df_sorted['sentimento_lag1'] = df_sorted['sentimento'].shift(1)
    
    # Remover NaN do lag
    dados_autocorr = df_sorted.dropna(subset=['sentimento_lag1'])
    
    # Calcular autocorrelação
    autocorr = df_sorted['sentimento'].autocorr(lag=1)
    
    fig.add_trace(
        go.Scatter(
            x=dados_autocorr['sentimento_lag1'],
            y=dados_autocorr['sentimento'],
            mode='markers',
            name='Autocorrelação',
            marker=dict(
                size=8,
                color='purple',
                opacity=0.6
            ),
            hovertemplate='<b>Sentimento Anterior:</b> %{x:.3f}<br><b>Sentimento Atual:</b> %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Adicionar linha de tendência
    if len(dados_autocorr) > 1:
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(dados_autocorr['sentimento_lag1'], dados_autocorr['sentimento'])
        
        x_line = np.array([dados_autocorr['sentimento_lag1'].min(), dados_autocorr['sentimento_lag1'].max()])
        y_line = slope * x_line + intercept
        
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=f'Tendência (R²={r_value**2:.3f})',
                line=dict(color='red', dash='dash', width=2),
                hovertemplate='<b>Linha de Tendência</b><br>R²: %{text}<extra></extra>',
                text=[f'{r_value**2:.3f}'] * len(x_line)
            ),
            row=2, col=1
        )
    
    # === 4. ROLLING STATISTICS COM BANDAS ===
    window = 6
    df_sorted['rolling_mean'] = df_sorted['sentimento'].rolling(window).mean()
    df_sorted['rolling_std'] = df_sorted['sentimento'].rolling(window).std()
    
    # Bandas de confiança
    upper_band = df_sorted['rolling_mean'] + df_sorted['rolling_std']
    lower_band = df_sorted['rolling_mean'] - df_sorted['rolling_std']
    
    # Banda de confiança (fill between)
    fig.add_trace(
        go.Scatter(
            x=df_sorted['DataReferencia'],
            y=upper_band,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_sorted['DataReferencia'],
            y=lower_band,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(width=0),
            name='±1 Desvio Padrão',
            hovertemplate='<b>Data:</b> %{x}<br><b>Limite:</b> %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Média móvel
    fig.add_trace(
        go.Scatter(
            x=df_sorted['DataReferencia'],
            y=df_sorted['rolling_mean'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Média Móvel',
            hovertemplate='<b>Data:</b> %{x}<br><b>Média Móvel:</b> %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Pontos originais
    fig.add_trace(
        go.Scatter(
            x=df_sorted['DataReferencia'],
            y=df_sorted['sentimento'],
            mode='markers',
            marker=dict(
                size=4,
                color='darkblue',
                opacity=0.5
            ),
            name='Sentimento',
            hovertemplate='<b>Data:</b> %{x}<br><b>Sentimento:</b> %{y:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # === ATUALIZAR LAYOUT ===
    fig.update_layout(
        height=800,
        title={
            'text': 'Análise de Eventos Extremos - COPOM',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        showlegend=True,
        template='plotly_white'
    )
    
    # Atualizar eixos específicos
    # Timeline
    fig.update_xaxes(title_text="Data", showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Sentimento", showgrid=True, row=1, col=1)
    
    # Lollipop (usar labels_datas da seção anterior)
    fig.update_xaxes(title_text="Score de Sentimento", showgrid=True, row=1, col=2)
    fig.update_yaxes(
        title_text="Período", 
        tickmode='array',
        tickvals=list(range(len(extremos))),
        ticktext=labels_datas,
        row=1, col=2
    )
    
    # Autocorrelação
    fig.update_xaxes(title_text="Sentimento Anterior", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Sentimento Atual", showgrid=True, row=2, col=1)
    
    # Rolling statistics
    fig.update_xaxes(title_text="Data", showgrid=True, row=2, col=2)
    fig.update_yaxes(title_text="Sentimento", showgrid=True, row=2, col=2)
    
    # Adicionar anotação da autocorrelação
    fig.add_annotation(
        text=f'Autocorrelação: {autocorr:.3f}',
        xref="x3", yref="y3",
        x=0.05, y=0.95,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="wheat",
        bordercolor="black",
        borderwidth=1,
        row=2, col=1
    )
    
    return fig


#################################################################################################
# Função para plotar sentimento vs selic vs IPCA
#################################################################################################

def plot_evolucao_temporal(df_merged):
    """Gráfico de evolução temporal: Sentimento vs Selic vs IPCA"""
    
    df = df_merged.dropna().copy()
    
    # Criar gráfico com eixo Y secundário
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Sentimento (eixo Y primário)
    fig.add_trace(
        go.Scatter(
            x=df['Data'],
            y=df['sentimento'],
            mode='lines+markers',
            name='Sentimento COPOM',
            line=dict(color='blue', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Data:</b> %{x}<br><b>Sentimento:</b> %{y:.3f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Selic (eixo Y secundário)
    fig.add_trace(
        go.Scatter(
            x=df['Data'],
            y=df['Selic'],
            mode='lines+markers',
            name='Taxa Selic',
            line=dict(color='red', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Data:</b> %{x}<br><b>Selic:</b> %{y:.2f}%<extra></extra>'
        ),
        secondary_y=True
    )
    
    # IPCA como área sombreada (normalizada para escala do sentimento)
    if 'IPCA' in df.columns and not df['IPCA'].isna().all():
        # Normalizar IPCA para a escala do sentimento
        ipca_min, ipca_max = df['IPCA'].min(), df['IPCA'].max()
        sent_min, sent_max = df['sentimento'].min(), df['sentimento'].max()
        
        if ipca_max != ipca_min:  # Evitar divisão por zero
            ipca_norm = (df['IPCA'] - ipca_min) / (ipca_max - ipca_min)
            ipca_scaled = ipca_norm * (sent_max - sent_min) + sent_min
            
            fig.add_trace(
                go.Scatter(
                    x=df['Data'],
                    y=ipca_scaled,
                    mode='lines',
                    name=f'IPCA (escala ajustada)',
                    line=dict(color='green', width=2, dash='dot'),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,0,0.1)',
                    hovertemplate='<b>Data:</b> %{x}<br><b>IPCA:</b> %{customdata:.2f}%<extra></extra>',
                    customdata=df['IPCA']
                ),
                secondary_y=False
            )
    
    # Adicionar linhas de referência
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=False)
    
    # Configurar eixos
    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text="Score de Sentimento", title_font_color="blue", secondary_y=False)
    fig.update_yaxes(title_text="Taxa Selic (%)", title_font_color="red", secondary_y=True)
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Evolução Temporal: Sentimento COPOM vs Taxa Selic vs IPCA',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


#################################################################################################
# Função para plotar correlações e lags
#################################################################################################


def plot_correlacoes_e_lags(df_merged, max_lags=12):
    """
    Análise combinada: Correlações (com R²) e Lags
    
    Layout 2x2:
    - Top Left: Scatter Sentimento vs Selic (com R²)
    - Top Right: Scatter Sentimento vs IPCA (com R²)
    - Bottom Left: Lags Sentimento vs Selic
    - Bottom Right: Lags Sentimento vs IPCA
    """
    
    df = df_merged.dropna().copy()
    
    if len(df) < 10:
        raise ValueError("Poucos dados disponíveis após remoção de NaN.")
    
    # Criar subplots 2x2
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Correlação: Sentimento vs Selic',
            'Correlação: Sentimento vs IPCA',
            'Análise de Lags: Sentimento vs Selic',
            'Análise de Lags: Sentimento vs IPCA'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # === 1. SCATTER SENTIMENTO VS SELIC ===
    corr_sent_selic = df['sentimento'].corr(df['Selic'])
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['sentimento'],
            y=df['Selic'],
            mode='markers',
            name='Sent vs Selic',
            marker=dict(
                color=df['IPCA'] if 'IPCA' in df.columns else 'steelblue',
                colorscale='Viridis',
                size=8,
                opacity=0.7,
                colorbar=dict(
                    title="IPCA %",
                    x=0.48,
                    len=0.4,
                    y=0.75
                ),
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Data:</b> %{customdata}<br><b>Sentimento:</b> %{x:.3f}<br><b>Selic:</b> %{y:.2f}%<br><b>IPCA:</b> %{marker.color:.2f}%<extra></extra>',
            customdata=df['Data'].dt.strftime('%Y-%m'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Linha de tendência Sentimento vs Selic
    if len(df) > 1:
        slope_ss, intercept_ss, r_ss, _, _ = stats.linregress(df['sentimento'], df['Selic'])
        x_line_ss = np.array([df['sentimento'].min(), df['sentimento'].max()])
        y_line_ss = slope_ss * x_line_ss + intercept_ss
        
        fig.add_trace(
            go.Scatter(
                x=x_line_ss,
                y=y_line_ss,
                mode='lines',
                name=f'Trend S-S (R²={r_ss**2:.3f})',
                line=dict(color='red', width=3, dash='dash'),
                hovertemplate=f'<b>Linha de Tendência</b><br>R² = {r_ss**2:.3f}<br>Correlação = {corr_sent_selic:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # === 2. SCATTER SENTIMENTO VS IPCA ===
    corr_sent_ipca = df['sentimento'].corr(df['IPCA'])
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['sentimento'],
            y=df['IPCA'],
            mode='markers',
            name='Sent vs IPCA',
            marker=dict(
                color=df['Selic'],
                colorscale='Plasma',
                size=8,
                opacity=0.7,
                colorbar=dict(
                    title="Selic %",
                    x=0.98,
                    len=0.4,
                    y=0.75
                ),
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>Data:</b> %{customdata}<br><b>Sentimento:</b> %{x:.3f}<br><b>IPCA:</b> %{y:.2f}%<br><b>Selic:</b> %{marker.color:.2f}%<extra></extra>',
            customdata=df['Data'].dt.strftime('%Y-%m'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Linha de tendência Sentimento vs IPCA
    if len(df) > 1:
        slope_si, intercept_si, r_si, _, _ = stats.linregress(df['sentimento'], df['IPCA'])
        x_line_si = np.array([df['sentimento'].min(), df['sentimento'].max()])
        y_line_si = slope_si * x_line_si + intercept_si
        
        fig.add_trace(
            go.Scatter(
                x=x_line_si,
                y=y_line_si,
                mode='lines',
                name=f'Trend S-I (R²={r_si**2:.3f})',
                line=dict(color='green', width=3, dash='dash'),
                hovertemplate=f'<b>Linha de Tendência</b><br>R² = {r_si**2:.3f}<br>Correlação = {corr_sent_ipca:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # === 3. ANÁLISE DE LAGS - SENTIMENTO VS SELIC ===
    lags = range(-max_lags, max_lags + 1)
    correlations_selic = []
    
    for lag in lags:
        if lag == 0:
            corr = df['sentimento'].corr(df['Selic'])
        elif lag > 0:
            corr = df['sentimento'].corr(df['Selic'].shift(lag))
        else:
            corr = df['sentimento'].shift(-lag).corr(df['Selic'])
        correlations_selic.append(corr if not pd.isna(corr) else 0)
    
    # Cores baseadas na intensidade (Selic - tons de vermelho)
    cores_lag_selic = []
    for corr in correlations_selic:
        if abs(corr) > 0.4:
            cores_lag_selic.append('#FF0000' if corr > 0 else '#8B0000')  # Vermelho forte
        elif abs(corr) > 0.2:
            cores_lag_selic.append('#FFA500' if corr > 0 else '#FF8C00')  # Laranja
        else:
            cores_lag_selic.append('#C0C0C0')  # Cinza claro
    
    fig.add_trace(
        go.Bar(
            x=list(lags),
            y=correlations_selic,
            name='Corr S-Selic',
            marker_color=cores_lag_selic,
            hovertemplate='<b>Lag:</b> %{x} meses<br><b>Correlação:</b> %{y:.3f}<br>' +
                         '<b>Interpretação:</b> %{customdata}<extra></extra>',
            customdata=[
                f"Sentimento antecede Selic ({lag}m)" if lag > 0 
                else f"Selic antecede Sentimento ({abs(lag)}m)" if lag < 0 
                else "Contemporâneo" 
                for lag in lags
            ],
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Linha de referência Selic
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
    
    # === 4. ANÁLISE DE LAGS - SENTIMENTO VS IPCA ===
    correlations_ipca = []
    
    for lag in lags:
        if lag == 0:
            corr = df['sentimento'].corr(df['IPCA'])
        elif lag > 0:
            corr = df['sentimento'].corr(df['IPCA'].shift(lag))
        else:
            corr = df['sentimento'].shift(-lag).corr(df['IPCA'])
        correlations_ipca.append(corr if not pd.isna(corr) else 0)
    
    # Cores baseadas na intensidade (IPCA - tons de verde)
    cores_lag_ipca = []
    for corr in correlations_ipca:
        if abs(corr) > 0.4:
            cores_lag_ipca.append('#008000' if corr > 0 else '#006400')  # Verde forte
        elif abs(corr) > 0.2:
            cores_lag_ipca.append('#90EE90' if corr > 0 else '#228B22')  # Verde claro
        else:
            cores_lag_ipca.append('#C0C0C0')  # Cinza claro
    
    fig.add_trace(
        go.Bar(
            x=list(lags),
            y=correlations_ipca,
            name='Corr S-IPCA',
            marker_color=cores_lag_ipca,
            hovertemplate='<b>Lag:</b> %{x} meses<br><b>Correlação:</b> %{y:.3f}<br>' +
                         '<b>Interpretação:</b> %{customdata}<extra></extra>',
            customdata=[
                f"Sentimento antecede IPCA ({lag}m)" if lag > 0 
                else f"IPCA antecede Sentimento ({abs(lag)}m)" if lag < 0 
                else "Contemporâneo" 
                for lag in lags
            ],
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Linha de referência IPCA
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=2)
    
    # === IDENTIFICAR MELHORES LAGS ===
    best_lag_selic = lags[np.argmax(np.abs(correlations_selic))]
    best_corr_selic = max(correlations_selic, key=abs)
    
    best_lag_ipca = lags[np.argmax(np.abs(correlations_ipca))]
    best_corr_ipca = max(correlations_ipca, key=abs)
    
    # === ATUALIZAR LAYOUT ===
    fig.update_layout(
        height=800,
        title={
            'text': 'Análise de Correlações e Lags: Sentimento COPOM vs Selic vs IPCA',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial, sans-serif'}
        },
        showlegend=False,
        template='plotly_white'
    )
    
    # Configurar eixos específicos
    # Scatter plots
    fig.update_xaxes(title_text="Score de Sentimento", row=1, col=1)
    fig.update_yaxes(title_text="Taxa Selic (%)", row=1, col=1)
    
    fig.update_xaxes(title_text="Score de Sentimento", row=1, col=2)
    fig.update_yaxes(title_text="IPCA (%)", row=1, col=2)
    
    # Lags
    fig.update_xaxes(title_text="Lag (meses)", row=2, col=1)
    fig.update_yaxes(title_text="Correlação", row=2, col=1)
    
    fig.update_xaxes(title_text="Lag (meses)", row=2, col=2)
    fig.update_yaxes(title_text="Correlação", row=2, col=2)
    
    # Atualizar títulos dos subplots com informações de R² e melhor lag
    annotations = list(fig.layout.annotations)
    if len(annotations) >= 4:
        annotations[0]['text'] = f'Correlação: Sentimento vs Selic<br><sub>r = {corr_sent_selic:.3f} | R² = {r_ss**2:.3f}</sub>'
        annotations[1]['text'] = f'Correlação: Sentimento vs IPCA<br><sub>r = {corr_sent_ipca:.3f} | R² = {r_si**2:.3f}</sub>'
        annotations[2]['text'] = f'Lags: Sentimento vs Selic<br><sub>Melhor: {best_lag_selic}m (r = {best_corr_selic:.3f})</sub>'
        annotations[3]['text'] = f'Lags: Sentimento vs IPCA<br><sub>Melhor: {best_lag_ipca}m (r = {best_corr_ipca:.3f})</sub>'
    
    fig.layout.annotations = annotations
    
    # Retornar resultados
    results = {
        'correlations': {
            'sentimento_selic': corr_sent_selic,
            'sentimento_ipca': corr_sent_ipca,
            'r_squared': {
                'selic': r_ss**2,
                'ipca': r_si**2
            }
        },
        'best_lags': {
            'selic': best_lag_selic,
            'ipca': best_lag_ipca
        },
        'best_lag_correlations': {
            'selic': best_corr_selic,
            'ipca': best_corr_ipca
        },
        'all_lags': {
            'selic': dict(zip(lags, correlations_selic)),
            'ipca': dict(zip(lags, correlations_ipca))
        },
        'data_info': {
            'observations': len(df),
            'period': f"{df['Data'].min().strftime('%Y-%m')} a {df['Data'].max().strftime('%Y-%m')}"
        }
    }
    
    return fig, results

#################################################################################################
# Função para obter as estatisticas das correlações e lags
#################################################################################################

def criar_dataframes_correlacoes(results):
    """
    Converte os resultados de correlação em DataFrames organizados
    
    Args:
        results: dicionário com resultados da análise de correlações
        
    Returns:
        dict: dicionário com diferentes DataFrames
    """
    
    # 1. DataFrame principal - mais compacto e legível
    df_principal = pd.DataFrame({
        'Variavel': ['Selic', 'IPCA'],
        'Corr_Atual': [
            round(results['correlations']['sentimento_selic'], 4),
            round(results['correlations']['sentimento_ipca'], 4)
        ],
        'R2_Pct': [
            round(results['correlations']['r_squared']['selic'] * 100, 1),
            round(results['correlations']['r_squared']['ipca'] * 100, 1)
        ],
        'Melhor_Lag': [
            results['best_lags']['selic'],
            results['best_lags']['ipca']
        ],
        'Corr_Melhor_Lag': [
            round(results['best_lag_correlations']['selic'], 4),
            round(results['best_lag_correlations']['ipca'], 4)
        ],
        'Melhoria': [
            round(results['best_lag_correlations']['selic'] - results['correlations']['sentimento_selic'], 4),
            round(results['best_lag_correlations']['ipca'] - results['correlations']['sentimento_ipca'], 4)
        ]
    })
    
    # 2. DataFrame de todos os lags - formato longo
    all_lags_data = []
    
    # Para Selic
    for lag, corr in results['all_lags']['selic'].items():
        all_lags_data.append({
            'Variavel': 'Selic',
            'Lag': lag,
            'Correlacao': round(corr, 4),
            'Eh_Melhor': 'Sim' if lag == results['best_lags']['selic'] else 'Nao'
        })
    
    # Para IPCA
    for lag, corr in results['all_lags']['ipca'].items():
        all_lags_data.append({
            'Variavel': 'IPCA',
            'Lag': lag,
            'Correlacao': round(corr, 4),
            'Eh_Melhor': 'Sim' if lag == results['best_lags']['ipca'] else 'Nao'
        })
    
    df_todos_lags = pd.DataFrame(all_lags_data)
    
    # 3. DataFrame comparativo (formato wide)
    df_comparativo = pd.DataFrame({
        'Metrica': [
            'Correlacao_Atual',
            'R2_Pct',
            'Melhor_Lag',
            'Corr_Melhor_Lag',
            'Melhoria_Absoluta'
        ],
        'Selic': [
            f"{results['correlations']['sentimento_selic']:.4f}",
            f"{results['correlations']['r_squared']['selic']*100:.1f}%",
            f"{results['best_lags']['selic']} meses",
            f"{results['best_lag_correlations']['selic']:.4f}",
            f"{results['best_lag_correlations']['selic'] - results['correlations']['sentimento_selic']:+.4f}"
        ],
        'IPCA': [
            f"{results['correlations']['sentimento_ipca']:.4f}",
            f"{results['correlations']['r_squared']['ipca']*100:.1f}%",
            f"{results['best_lags']['ipca']} meses",
            f"{results['best_lag_correlations']['ipca']:.4f}",
            f"{results['best_lag_correlations']['ipca'] - results['correlations']['sentimento_ipca']:+.4f}"
        ]
    })
    
    # 4. DataFrame resumo
    melhor_var = 'IPCA' if abs(results['correlations']['sentimento_ipca']) > abs(results['correlations']['sentimento_selic']) else 'Selic'
    maior_r2 = 'IPCA' if results['correlations']['r_squared']['ipca'] > results['correlations']['r_squared']['selic'] else 'Selic'
    
    df_resumo = pd.DataFrame({
        'Aspecto': [
            'Periodo',
            'Observacoes',
            'Melhor_Correlacao',
            'Maior_R2',
            'Interpretacao_Lag_Selic',
            'Interpretacao_Lag_IPCA'
        ],
        'Valor': [
            results['data_info']['period'],
            f"{results['data_info']['observations']:,}",
            f"{melhor_var} ({results['correlations'][f'sentimento_{melhor_var.lower()}']:.4f})",
            f"{maior_r2} ({results['correlations']['r_squared'][maior_r2.lower()]*100:.1f}%)",
            f"Selic antecipa sentimento em {abs(results['best_lags']['selic'])} meses" if results['best_lags']['selic'] < 0 else f"Sentimento antecipa Selic em {results['best_lags']['selic']} meses",
            f"Sentimento antecipa IPCA em {results['best_lags']['ipca']} meses" if results['best_lags']['ipca'] > 0 else f"IPCA antecipa sentimento em {abs(results['best_lags']['ipca'])} meses"
        ]
    })
    
    return {
        'principal': df_principal,
        'todos_lags': df_todos_lags,
        'comparativo': df_comparativo,
        'resumo': df_resumo
    }

def exibir_correlacoes_dataframes(results):
    """Exibe os DataFrames de forma organizada"""
    dfs = criar_dataframes_correlacoes(results)
    
    print("="*80)
    print("CORRELAÇÕES EM DATAFRAMES".center(80))
    print("="*80)
    
    print("\n📊 RESUMO PRINCIPAL:")
    print(dfs['principal'].to_string(index=False))
    
    print("\n📈 COMPARATIVO:")
    print(dfs['comparativo'].to_string(index=False))
    
    print("\n📋 RESUMO EXECUTIVO:")
    print(dfs['resumo'].to_string(index=False))
    
    print("\n⏰ PRIMEIROS 10 LAGS DE CADA VARIÁVEL:")
    df_sample_selic = dfs['todos_lags'][dfs['todos_lags']['Variavel'] == 'Selic'].head(10)
    df_sample_ipca = dfs['todos_lags'][dfs['todos_lags']['Variavel'] == 'IPCA'].head(10)
    
    print("\nSelic:")
    print(df_sample_selic.to_string(index=False))
    print("\nIPCA:")
    print(df_sample_ipca.to_string(index=False))
    
    return dfs

def salvar_correlacoes_excel(results, nome_arquivo="correlacoes_analise.xlsx"):
    """Salva todos os DataFrames em Excel"""
    dfs = criar_dataframes_correlacoes(results)
    
    try:
        with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
            dfs['principal'].to_excel(writer, sheet_name='Principal', index=False)
            dfs['comparativo'].to_excel(writer, sheet_name='Comparativo', index=False)
            dfs['resumo'].to_excel(writer, sheet_name='Resumo', index=False)
            dfs['todos_lags'].to_excel(writer, sheet_name='Todos_Lags', index=False)
        
        print(f"✅ Arquivo salvo: {nome_arquivo}")
        return nome_arquivo
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        return None

# Função para uso rápido - retorna apenas o DataFrame principal
def correlacoes_df(results):
    """Retorna apenas o DataFrame principal para análise rápida"""
    return criar_dataframes_correlacoes(results)['principal']


#################################################################################################
# Função para obter o relatorio das estatisticas das correlações e lags
#################################################################################################

def relatorio_correlacoes_lags(df_merged, max_lags=12):
    """Relatório focado em correlações e lags"""

    # Gera o gráfico e resultados
    fig, results = plot_correlacoes_e_lags(df_merged, max_lags)
    
    # Exibe o relatório textual
    print("="*70)
    print("RELATÓRIO: CORRELAÇÕES E ANÁLISE DE LAGS".center(70))
    print("="*70)
    
    print(f"\n📊 DADOS ANALISADOS:")
    print(f"   • Período: {results['data_info']['period']}")
    print(f"   • Observações: {results['data_info']['observations']}")
    
    print(f"\n📈 CORRELAÇÕES CONTEMPORÂNEAS:")
    corr = results['correlations']
    print(f"   • Sentimento vs Selic: {corr['sentimento_selic']:7.4f}")
    print(f"   • Sentimento vs IPCA:  {corr['sentimento_ipca']:7.4f}")
    
    print(f"\n🎯 PODER EXPLICATIVO (R²):")
    r2 = corr['r_squared']
    print(f"   • Sentimento explica Selic: {r2['selic']*100:5.1f}%")
    print(f"   • Sentimento explica IPCA:  {r2['ipca']*100:5.1f}%")
    
    print(f"\n⏰ MELHORES DEFASAGENS (LAGS):")
    lags = results['best_lags']
    lag_corr = results['best_lag_correlations']
    
    print(f"   • Sentimento vs Selic:")
    print(f"     - Melhor lag: {lags['selic']:3d} meses")
    print(f"     - Correlação: {lag_corr['selic']:7.4f}")
    if lags['selic'] > 0:
        print(f"     - Sentimento ANTECEDE Selic em {lags['selic']} meses")
    elif lags['selic'] < 0:
        print(f"     - Selic ANTECEDE Sentimento em {abs(lags['selic'])} meses")
    else:
        print(f"     - Relação CONTEMPORÂNEA")
    
    print(f"   • Sentimento vs IPCA:")
    print(f"     - Melhor lag: {lags['ipca']:3d} meses")
    print(f"     - Correlação: {lag_corr['ipca']:7.4f}")
    if lags['ipca'] > 0:
        print(f"     - Sentimento ANTECEDE IPCA em {lags['ipca']} meses")
    elif lags['ipca'] < 0:
        print(f"     - IPCA ANTECEDE Sentimento em {abs(lags['ipca'])} meses")
    else:
        print(f"     - Relação CONTEMPORÂNEA")
    
    print(f"\n💡 INTERPRETAÇÃO ECONÔMICA:")
    
    # Análises mais detalhadas baseadas nos resultados
    if abs(corr['sentimento_selic']) > 0.5:
        print(f"   ✓ FORTE relação Sentimento-Selic ({corr['sentimento_selic']:.3f})")
    elif abs(corr['sentimento_selic']) > 0.3:
        print(f"   ⚠ MODERADA relação Sentimento-Selic ({corr['sentimento_selic']:.3f})")
    else:
        print(f"   ⚠ FRACA relação Sentimento-Selic ({corr['sentimento_selic']:.3f})")
        
    if abs(corr['sentimento_ipca']) > 0.5:
        print(f"   ✓ FORTE relação Sentimento-IPCA ({corr['sentimento_ipca']:.3f})")
    elif abs(corr['sentimento_ipca']) > 0.3:
        print(f"   ✓ MODERADA relação Sentimento-IPCA ({corr['sentimento_ipca']:.3f})")
    else:
        print(f"   ⚠ FRACA relação Sentimento-IPCA ({corr['sentimento_ipca']:.3f})")
    
    # Interpretações sobre os lags
    if lags['selic'] > 0:
        print(f"   ✓ COPOM sinaliza mudanças na Selic com {lags['selic']} meses de antecedência")
    elif lags['selic'] < 0:
        print(f"   ✓ Sentimento reage às mudanças da Selic com {abs(lags['selic'])} meses de defasagem")
        
    if lags['ipca'] > 0:
        print(f"   ✓ COPOM antecipa pressões inflacionárias com {lags['ipca']} meses de antecedência")
    elif lags['ipca'] < 0:
        print(f"   ✓ Sentimento reage ao IPCA com {abs(lags['ipca'])} meses de defasagem")
    
    # Análise do poder explicativo
    if r2['selic'] > 0.25:
        print(f"   ✓ Sentimento explica {r2['selic']*100:.1f}% da variação da Selic")
    elif r2['selic'] > 0.10:
        print(f"   ⚠ Sentimento explica apenas {r2['selic']*100:.1f}% da variação da Selic")
    else:
        print(f"   ⚠ Baixo poder explicativo: {r2['selic']*100:.1f}% da variação da Selic")
        
    if r2['ipca'] > 0.25:
        print(f"   ✓ Sentimento explica {r2['ipca']*100:.1f}% da variação do IPCA")
    elif r2['ipca'] > 0.10:
        print(f"   ⚠ Sentimento explica apenas {r2['ipca']*100:.1f}% da variação do IPCA")
    else:
        print(f"   ⚠ Baixo poder explicativo: {r2['ipca']*100:.1f}% da variação do IPCA")
    
    print("\n🔍 RESUMO EXECUTIVO:")
    print(f"   • Correlações são FRACAS/MODERADAS (< 0.3)")
    print(f"   • Poder explicativo é BAIXO (< 10%)")
    print(f"   • Evidência de que sentimento ANTECIPA IPCA em {lags['ipca']} meses")
    print(f"   • Evidência de que Selic ANTECIPA sentimento em {abs(lags['selic'])} meses")
    print(f"   • Sugere que COPOM considera expectativas inflacionárias futuras")
    
    print("="*70)
    
    return results





#################################################################################################
# === 8. PLOTAR SENTIMENTO COPOM ===
#################################################################################################

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

#################################################################################################
# === 9. PLOTAR SENTIMENTO COPOM COM PRESIDENTES ===
#################################################################################################

def plotar_sentimento_copom_pres(sentimento, titulo="Evolução do Sentimento nas Atas do COPOM"):
    """
    Plota a evolução temporal do sentimento nas atas do COPOM com estilo aprimorado,
    cores vibrantes, e marcações para as mudanças de presidentes do BCB.
    
    Args:
        sentimento: DataFrame com colunas 'DataReferencia' e 'sentimento'
        titulo: Título do gráfico
    
    Retorna:
        matplotlib.figure.Figure: objeto da figura (para salvar ou customizar)
    """

    if sentimento is None or sentimento.empty:
        print("Erro: DataFrame de sentimento está vazio ou é None")
        return None

    if 'DataReferencia' not in sentimento.columns or 'sentimento' not in sentimento.columns:
        print("Erro: Colunas 'DataReferencia' e/ou 'sentimento' não encontradas")
        print(f"Colunas disponíveis: {sentimento.columns.tolist()}")
        return None

    try:
        sentimento = sentimento.sort_values("DataReferencia").copy()
        sentimento = sentimento.dropna(subset=['DataReferencia', 'sentimento'])

        # Ajustar timezone e cortar dados
        if hasattr(sentimento['DataReferencia'].dtype, 'tz') and sentimento['DataReferencia'].dtype.tz is not None:
            timezone = sentimento['DataReferencia'].dtype.tz
            cutoff_date = pd.Timestamp('2012-01-01').tz_localize(timezone)
        else:
            cutoff_date = pd.Timestamp('2012-01-01')

        sentimento = sentimento[sentimento['DataReferencia'] >= cutoff_date]
        if sentimento.empty:
            print("Erro: Não há dados válidos após remoção de valores nulos e filtro para 2012+")
            return None

        # Transições de presidentes
        if hasattr(sentimento['DataReferencia'].dtype, 'tz') and sentimento['DataReferencia'].dtype.tz is not None:
            timezone = sentimento['DataReferencia'].dtype.tz
            transicoes_presidentes = [
                {'data': pd.Timestamp('2016-06-09').tz_localize(timezone), 'saida': 'Alexandre Tombini', 'entrada': 'Ilan Goldfajn'},
                {'data': pd.Timestamp('2019-02-28').tz_localize(timezone), 'saida': 'Ilan Goldfajn', 'entrada': 'Roberto Campos Neto'},
                {'data': pd.Timestamp('2025-01-01').tz_localize(timezone), 'saida': 'Roberto Campos Neto', 'entrada': 'Gabriel Galípolo'}
            ]
        else:
            transicoes_presidentes = [
                {'data': pd.Timestamp('2016-06-09'), 'saida': 'Alexandre Tombini', 'entrada': 'Ilan Goldfajn'},
                {'data': pd.Timestamp('2019-02-28'), 'saida': 'Ilan Goldfajn', 'entrada': 'Roberto Campos Neto'},
                {'data': pd.Timestamp('2025-01-01'), 'saida': 'Roberto Campos Neto', 'entrada': 'Gabriel Galípolo'}
            ]

        # Paleta de cores vibrantes e com contraste
        cor_pos = "#008000"       # azul forte
        cor_neg = "#FF0000"       # vermelho forte
        cor_linha = "#000000"     # preto
        cor_pontos = "#010000"    # verde médio
        cor_transicao = "#ff7f0e" # laranja
        cor_fundo_anotacao = "#ffffff"  # branco puro

        # Início do plot
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_facecolor("#f5f5f5")  # fundo levemente acinzentado

        # Área positiva
        ax.fill_between(sentimento["DataReferencia"], 0, sentimento["sentimento"],
                        where=sentimento["sentimento"] > 0, interpolate=True,
                        color=cor_pos, alpha=0.7)

        # Área negativa
        ax.fill_between(sentimento["DataReferencia"], 0, sentimento["sentimento"],
                        where=sentimento["sentimento"] < 0, interpolate=True,
                        color=cor_neg, alpha=0.7)

        # Linha principal
        ax.plot(sentimento["DataReferencia"], sentimento["sentimento"],
                color=cor_linha, linewidth=2.5, zorder=10)

        # Pontos
        ax.scatter(sentimento["DataReferencia"], sentimento["sentimento"],
                   color=cor_pontos, s=45, alpha=1.0, zorder=15,
                   edgecolors='black', linewidth=0.6)

        # Linha horizontal zero
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1.5)

        data_min = sentimento["DataReferencia"].min()
        data_max = sentimento["DataReferencia"].max()

        # Presidente inicial
        presidente_inicial = "Alexandre Tombini"
        for transicao in transicoes_presidentes:
            if transicao['data'] <= data_min:
                presidente_inicial = transicao['entrada']

        y_max = sentimento['sentimento'].max()
        y_min = sentimento['sentimento'].min()
        range_y = y_max - y_min

        ax.text(data_min + pd.Timedelta(days=100), y_min - range_y * 0.1,
                f"{presidente_inicial}", fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=cor_fundo_anotacao,
                          edgecolor="black", alpha=1.0),
                ha='left', va='top')

        for i, transicao in enumerate(transicoes_presidentes):
            if data_min <= transicao['data'] <= data_max:
                ax.axvline(x=transicao['data'], color=cor_transicao,
                           linestyle='--', alpha=1.0, linewidth=2.5, zorder=5)
                y_pos = y_min - range_y * (0.15 + i * 0.12)
                ax.text(transicao['data'], y_pos,
                        f"{transicao['entrada']}\n{transicao['data'].strftime('%m/%Y')}",
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=cor_fundo_anotacao,
                                  edgecolor='black', alpha=1.0),
                        ha='left', va='top')

        num_transicoes = len([t for t in transicoes_presidentes if data_min <= t['data'] <= data_max])
        if num_transicoes > 0:
            ax.set_ylim(y_min - range_y * (0.3 + num_transicoes * 0.12), y_max + range_y * 0.05)

        # Títulos
        ax.set_title(titulo, fontsize=18, weight="bold", loc="center", pad=25)
        fig.suptitle("Análise Temporal (2012+) | Mudanças de Presidência do BCB",
                     fontsize=13, y=0.96, style='italic')
        ax.set_xlabel("Data da Reunião", fontsize=13, weight='bold')
        ax.set_ylabel("Índice de Sentimento", fontsize=13, weight='bold')

        # Eixo x
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.8)
        ax.grid(axis='x', linestyle=':', alpha=0.4, linewidth=0.6)

        # Legenda estilizada
        legend_elements = [
            plt.Line2D([0], [0], color=cor_pos, alpha=1.0, linewidth=10, label='Sentimento Positivo'),
            plt.Line2D([0], [0], color=cor_neg, alpha=1.0, linewidth=10, label='Sentimento Negativo'),
            plt.Line2D([0], [0], color=cor_linha, linewidth=2.5, label='Evolução do Sentimento'),
            plt.Line2D([0], [0], color=cor_transicao, linestyle='--', linewidth=2.5, label='Mudança de Presidente BCB')
        ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
                  title="Legenda", title_fontsize=12, frameon=True,
                  fancybox=True, framealpha=0.9)

        # Estilo dos eixos
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')

        fig.text(0.99, 0.01, "Dados: BCB | Transições de Presidência BCB destacadas",
                 ha='right', fontsize=10, style='italic')

        plt.tight_layout(rect=[0, 0.05, 1, 0.98])

        print(f"Gráfico criado com sucesso! Período: {sentimento['DataReferencia'].min()} a {sentimento['DataReferencia'].max()}")
        print(f"Número de observações: {len(sentimento)}")
        print(f"Transições de presidência no período: {num_transicoes}")

        return fig

    except Exception as e:
        print(f"Erro ao criar o gráfico: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


#################################################################################################
# Função auxiliar para obter apenas os valores dos sentimentos em um df
#################################################################################################

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

    return df_com_meses_completos


