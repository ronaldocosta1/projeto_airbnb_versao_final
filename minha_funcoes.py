# Importações necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Matriz de Correlação das Variáveis Numéricas",
    ax: Optional[plt.Axes] = None,
    cmap: str = 'coolwarm',
    fmt: str = '.2f',
    linewidths: float = 0.5,
    save_path: Optional[str] = None,
    save_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Gera e exibe um heatmap de correlação aprimorado e mais informativo,
    com opção para exportação em alta qualidade.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("O parâmetro 'df' deve ser um pandas DataFrame.")

    corr_matrix = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        linewidths=linewidths,
        ax=ax,
        cbar_kws={'label': 'Nível de Correlação'}
    )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='y', rotation=0)

    # --- LINHA CORRIGIDA ---
    # Removemos o argumento 'ha' que não é válido para esta função.
    # A rotação dos rótulos já melhora a legibilidade.
    ax.tick_params(axis='x', rotation=45)

    if created_fig:
        plt.tight_layout(pad=1.5)

    if save_path:
        default_save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.1}
        if save_kwargs:
            default_save_kwargs.update(save_kwargs)
        print(f"Salvando o heatmap em '{save_path}' com os parâmetros: {default_save_kwargs}")
        fig.savefig(save_path, **default_save_kwargs)

    if created_fig:
        plt.show()
        
        
        
        
        
        
        
        
        
        
# Importações necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Optional
import locale

def plot_boxplot_informativo_reais(
    data: pd.Series,
    whis: float = 1.5,
    output_path: Optional[str] = None,
    dpi: int = 300,
    format_as_currency: bool = False,
    titulo: Optional[str] = None, # NOVO: Parâmetro para o título principal
    nome_eixo_x: Optional[str] = None # NOVO: Parâmetro para o rótulo do eixo X
):
    """
    Gera e, opcionalmente, salva dois boxplots lado a lado com ricas anotações.

    Args:
        data (pd.Series): A série de dados a ser plotada.
        whis (float, optional): Fator do IQR para definir os limites. Padrão 1.5.
        output_path (str, optional): Caminho para salvar o gráfico.
        dpi (int, optional): Resolução em DPI para salvar em PNG/JPG.
        format_as_currency (bool, optional): Se True, formata os valores como moeda (R$).
        titulo (str, optional): Título customizado para o gráfico.
        nome_eixo_x (str, optional): Rótulo customizado para o eixo X.
    """
    # Validação da entrada
    if not isinstance(data, pd.Series):
        raise TypeError("O parâmetro 'data' deve ser uma pandas Series.")

    # Lógica de Formatação (sem alterações)
    if format_as_currency:
        try:
            locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
            FMT_CURRENCY = lambda val: locale.currency(val, symbol=True, grouping=True)
            FMT_INT = lambda val: locale.format_string("%d", val, grouping=True)
            FMT_PCT = lambda val: locale.format_string("%.2f", val, grouping=True)
            axis_formatter = FuncFormatter(lambda x, p: locale.currency(x, symbol=False, grouping=True))
        except locale.Error:
            print("Aviso: Locale 'pt_BR.UTF-8' não encontrado. Usando formatação manual.")
            FMT_CURRENCY = lambda val: f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            FMT_INT = lambda val: f"{val:,}".replace(',', '.')
            FMT_PCT = lambda val: f"{val:.2f}".replace('.', ',')
            axis_formatter = FuncFormatter(lambda x, p: f"{x/1000:.0f}k")
    else:
        FMT_CURRENCY = lambda val: f"{val:,.2f}"
        FMT_INT = lambda val: f"{val:,}"
        FMT_PCT = lambda val: f"{val:.2f}"
        axis_formatter = None

    # Cálculos Estatísticos (sem alterações)
    q1 = data.quantile(0.25)
    median = data.median()
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    mean = data.mean()
    lower_bound = q1 - whis * iqr
    upper_bound = q3 + whis * iqr
    n = len(data)
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / n) * 100

    # Configuração da Figura
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # ALTERADO: Define o título principal usando o parâmetro customizado ou um padrão
    titulo_final = titulo if titulo else f'Análise de Distribuição e Outliers para "{data.name}"'
    fig.suptitle(titulo_final, fontsize=18, fontweight='bold')
    
    # ALTERADO: Define o rótulo do eixo X usando o parâmetro customizado ou um padrão
    xlabel_final = nome_eixo_x if nome_eixo_x else f'Valores de {data.name}'

    # Gráfico da Esquerda: Visão Completa
    sns.boxplot(x=data, ax=axes[0], whis=whis, color='#6baed6')
    axes[0].set_title('Distribuição Completa (com Outliers)', fontsize=14)
    axes[0].set_xlabel(xlabel_final) # Usa o rótulo final
    axes[0].axvline(median, color='orange', linestyle='--', linewidth=2, label=f'Mediana ({FMT_CURRENCY(median)})')
    axes[0].axvline(mean, color='red', linestyle=':', linewidth=2, label=f'Média ({FMT_CURRENCY(mean)})')
    axes[0].legend()

    # Gráfico da Direita: Visão Ampliada e Informativa
    sns.boxplot(x=data, ax=axes[1], whis=whis, color='#74c476')
    axes[1].set_xlim(lower_bound, upper_bound)
    axes[1].set_title('Visão Ampliada (Foco na Distribuição Principal)', fontsize=14)
    axes[1].set_xlabel(xlabel_final) # Usa o rótulo final
    axes[1].axvspan(q1, q3, alpha=0.2, color='yellow', label=f'IQR ({FMT_CURRENCY(iqr)})')
    axes[1].axvline(median, color='orange', linestyle='--', linewidth=2, label=f'Mediana ({FMT_CURRENCY(median)})')
    axes[1].axvline(mean, color='red', linestyle=':', linewidth=2, label=f'Média ({FMT_CURRENCY(mean)})')
    axes[1].legend()

    stats_text = ( f"Estatísticas Descritivas:\n" f"--------------------------\n" f"Contagem Total (n): {FMT_INT(n)}\n" f"Média: {FMT_CURRENCY(mean)}\n" f"Mediana (Q2): {FMT_CURRENCY(median)}\n" f"Q1 (25%): {FMT_CURRENCY(q1)}\n" f"Q3 (75%): {FMT_CURRENCY(q3)}\n" f"IQR (Q3-Q1): {FMT_CURRENCY(iqr)}\n" f"--------------------------\n" f"Análise de Outliers:\n" f"--------------------------\n" f"Limite Inferior: {FMT_CURRENCY(lower_bound)}\n" f"Limite Superior: {FMT_CURRENCY(upper_bound)}\n" f"Nº de Outliers: {FMT_INT(n_outliers)} ({FMT_PCT(pct_outliers)}%)" )
    axes[1].text(0.03, 0.97, stats_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.8))

    if format_as_currency and axis_formatter:
        axes[0].xaxis.set_major_formatter(axis_formatter)
        axes[1].xaxis.set_major_formatter(axis_formatter)

    # Finalização, Exportação e Exibição
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Gráfico salvo com sucesso em: {output_path}")
        plt.close(fig)
    else:
        plt.show()
        
        
        
        
        
        
        
        
        
# Importações necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing import Optional, Dict, Any

# Função auxiliar para formatação inteligente do eixo Y
def human_readable_formatter(x, pos):
    """Formata os números do eixo em um formato legível (ex: 120000 -> 120k)."""
    if x >= 1_000_000:
        s = f'{x/1_000_000:.1f}'.replace('.', ',')
        return s.replace(',0', '') + 'M'
    if x >= 1_000:
        s = f'{x/1_000:.1f}'.replace('.', ',')
        return s.replace(',0', '') + 'k'
    return int(x)


def plot_histogram_reais(
    data_series: pd.Series,
    ax: Optional[plt.Axes] = None,
    bins: int = 30,
    kde: bool = True,
    title: Optional[str] = None,
    color: str = "#3498db",
    alpha: float = 0.7,
    style: str = 'whitegrid',
    show_mean: bool = False,
    show_median: bool = False,
    show_perc_on_bar: bool = False,
    tick_labelsize: int = 12,
    save_path: Optional[str] = None,
    save_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Plota um histograma completo com formatação de moeda (R$), decimais com vírgula,
    eixos inteligentes e opção de exportação em alta qualidade.
    """
    if not isinstance(data_series, pd.Series):
        raise TypeError("O parâmetro 'data_series' deve ser uma pandas Series.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    with sns.axes_style(style):
        sns.histplot(data=data_series, bins=bins, kde=kde, ax=ax, color=color, alpha=alpha, edgecolor='white', linewidth=0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_formatter))

        # --- ALTERAÇÃO AQUI: Inicializa a soma da porcentagem ---
        total_displayed_perc = 0.0

        if show_perc_on_bar:
            total = len(data_series)
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    percentage_val = (height / total) * 100
                    # --- ALTERAÇÃO AQUI: Acumula a porcentagem ---
                    total_displayed_perc += percentage_val
                    percentage_str = f'{percentage_val:.1f}'.replace('.', ',') + '%'
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ax.annotate(text=percentage_str, xy=(x, y), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=11, color='dimgray', fontweight='bold')

        final_title = title if title else f'Distribuição de {data_series.name}'
        ax.set_title(final_title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel(data_series.name, fontsize=14, labelpad=15)
        ax.set_ylabel('Frequência', fontsize=14, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

        # --- ALTERAÇÃO AQUI: Lógica de legenda refeita para maior flexibilidade ---
        handles, labels = [], []

        # Adiciona o Total Exibido primeiro, se aplicável
        if show_perc_on_bar:
            total_label = f'Total Exibido: {total_displayed_perc:.1f}'.replace('.', ',') + '%'
            # Adiciona um "handle" invisível para alinhar o texto na legenda
            handles.append(Rectangle((0,0), 0, 0, fill=False, edgecolor='none', visible=False))
            labels.append(total_label)

        if show_mean:
            mean_val = data_series.mean()
            label = f'Média: R$ {f"{mean_val:.2f}".replace(".", ",")}'
            ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2.5)
            handles.append(Line2D([0], [0], color='#e74c3c', lw=2.5, linestyle='--'))
            labels.append(label)

        if show_median:
            median_val = data_series.median()
            label = f'Mediana: R$ {f"{median_val:.2f}".replace(".", ",")}'
            ax.axvline(median_val, color='#9b59b6', linestyle=':', linewidth=2.5)
            handles.append(Line2D([0], [0], color='#9b59b6', lw=2.5, linestyle=':'))
            labels.append(label)

        if handles:
            ax.legend(handles, labels, fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=False, borderpad=1)

        sns.despine(ax=ax)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    if created_fig:
        plt.tight_layout()
    if save_path:
        default_save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.1}
        if save_kwargs:
            default_save_kwargs.update(save_kwargs)
        print(f"Salvando o gráfico em '{save_path}' com os parâmetros: {default_save_kwargs}")
        fig.savefig(save_path, **default_save_kwargs)
    if created_fig:
        plt.show()
        
        
        
        
def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5*amplitude, q3 + 1.5*amplitude

def excluir_outliers(df, nome_coluna):
    quantidade_de_linhas = df.shape[0] # quantidade de linhas antes de excluir
    lim_inf, lim_sup = limites(df[nome_coluna]) # limites
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup),:] # todas as linhas maiores do que o limite inferior e menores do que o limite superior
    linhas_removidas = quantidade_de_linhas - df.shape[0]
    return df, linhas_removidas









# Importações necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator # <--- ADICIONEI MaxNLocator AQUI
from typing import Optional
import locale

def plot_boxplot_informativo(
    data: pd.Series,
    whis: float = 1.5,
    output_path: Optional[str] = None,
    dpi: int = 300,
    format_as_currency: bool = False,
    titulo: Optional[str] = None,
    nome_eixo_x: Optional[str] = None
):
    """
    Gera e, opcionalmente, salva dois boxplots lado a lado com ricas anotações.

    Args:
        data (pd.Series): A série de dados a ser plotada.
        whis (float, optional): Fator do IQR para definir os limites. Padrão 1.5.
        output_path (str, optional): Caminho para salvar o gráfico.
        dpi (int, optional): Resolução em DPI para salvar em PNG/JPG.
        format_as_currency (bool, optional): Se True, formata os valores como moeda (R$).
        titulo (str, optional): Título customizado para o gráfico.
        nome_eixo_x (str, optional): Rótulo customizado para o eixo X.
    """
    # Validação da entrada
    if not isinstance(data, pd.Series):
        raise TypeError("O parâmetro 'data' deve ser uma pandas Series.")

    # Lógica de Formatação
    try:
        locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
        FMT_INT = lambda val: locale.format_string("%d", val, grouping=True)
        FMT_NUM = lambda val: locale.format_string("%.2f", val, grouping=True)
        FMT_PCT = lambda val: locale.format_string("%.2f", val, grouping=True) + '%'
        if format_as_currency:
            FMT_FINAL = lambda val: locale.currency(val, symbol=True, grouping=True)
        else:
            FMT_FINAL = FMT_NUM
        axis_formatter = FuncFormatter(lambda x, p: locale.format_string("%.0f", x, grouping=True))
    except locale.Error:
        print("Aviso: Locale 'pt_BR.UTF-8' não encontrado. Usando formatação manual.")
        FMT_INT = lambda val: f"{val:,}".replace(',', '.')
        FMT_NUM = lambda val: f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        FMT_PCT = lambda val: f"{val:.2f}".replace('.', ',') + '%'
        if format_as_currency:
            FMT_FINAL = lambda val: f"R$ {FMT_NUM(val)}"
        else:
            FMT_FINAL = FMT_NUM
        axis_formatter = FuncFormatter(lambda x, p: f"{int(x):,}".replace(',', '.'))

    # Cálculos Estatísticos
    q1 = data.quantile(0.25)
    median = data.median()
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    mean = data.mean()
    lower_bound = q1 - whis * iqr
    upper_bound = q3 + whis * iqr
    n = len(data)
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / n) * 100

    # Configuração da Figura
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    titulo_final = titulo if titulo else f'Análise de {data.name}'
    fig.suptitle(titulo_final, fontsize=18, fontweight='bold')
    
    xlabel_final = nome_eixo_x if nome_eixo_x else data.name

    # Gráfico da Esquerda: Visão Completa
    sns.boxplot(x=data, ax=axes[0], whis=whis, color='#6baed6')
    axes[0].set_title('Distribuição Completa (com Outliers)', fontsize=14)
    axes[0].set_xlabel(xlabel_final)
    axes[0].axvline(median, color='orange', linestyle='--', linewidth=2, label=f'Mediana ({FMT_FINAL(median)})')
    axes[0].axvline(mean, color='red', linestyle=':', linewidth=2, label=f'Média ({FMT_FINAL(mean)})')
    axes[0].legend()

    # Gráfico da Direita: Visão Ampliada e Informativa
    sns.boxplot(x=data, ax=axes[1], whis=whis, color='#74c476')
    axes[1].set_xlim(lower_bound, upper_bound)
    axes[1].set_title('Visão Ampliada (Foco na Distribuição Principal)', fontsize=14)
    axes[1].set_xlabel(xlabel_final)

    # --- CORREÇÃO AQUI ---
    # Força os marcadores (ticks) do eixo X a serem apenas inteiros.
    # Isso evita que o matplotlib crie ticks fracionários (ex: 0.2, 0.4) que,
    # após a formatação para 0 casas decimais, viram rótulos duplicados ("0", "0").
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    axes[1].axvspan(q1, q3, alpha=0.2, color='yellow', label=f'IQR ({FMT_FINAL(iqr)})')
    axes[1].axvline(median, color='orange', linestyle='--', linewidth=2, label=f'Mediana ({FMT_FINAL(median)})')
    axes[1].axvline(mean, color='red', linestyle=':', linewidth=2, label=f'Média ({FMT_FINAL(mean)})')
    axes[1].legend()

    # Texto de estatísticas
    stats_text = (
        f"Estatísticas Descritivas:\n"
        f"--------------------------\n"
        f"Contagem Total (n): {FMT_INT(n)}\n"
        f"Média: {FMT_FINAL(mean)}\n"
        f"Mediana (Q2): {FMT_FINAL(median)}\n"
        f"Q1 (25%): {FMT_FINAL(q1)}\n"
        f"Q3 (75%): {FMT_FINAL(q3)}\n"
        f"IQR (Q3-Q1): {FMT_FINAL(iqr)}\n"
        f"--------------------------\n"
        f"Análise de Outliers:\n"
        f"--------------------------\n"
        f"Limite Inferior: {FMT_FINAL(lower_bound)}\n"
        f"Limite Superior: {FMT_FINAL(upper_bound)}\n"
        f"Nº de Outliers: {FMT_INT(n_outliers)} ({FMT_PCT(pct_outliers)})"
    )
    
    axes[1].text(0.03, 0.97, stats_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.8))

    # Aplica formatação aos eixos
    if axis_formatter:
        axes[0].xaxis.set_major_formatter(axis_formatter)
        axes[1].xaxis.set_major_formatter(axis_formatter)

    # Finalização
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Gráfico salvo com sucesso em: {output_path}")
        plt.close(fig)
    else:
        plt.show()
        
        
        
        
        
        
        
        
        
        
 # Importações necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing import Optional, Dict, Any
import locale # <-- 1. IMPORTAÇÃO ADICIONADA

# Função auxiliar para formatação inteligente do eixo Y (não precisa de alteração)
def human_readable_formatter(x, pos):
    """Formata os números do eixo em um formato legível (ex: 120000 -> 120k)."""
    if x >= 1_000_000:
        s = f'{x/1_000_000:.1f}'.replace('.', ',')
        return s.replace(',0', '') + 'M'
    if x >= 1_000:
        s = f'{x/1_000:.1f}'.replace('.', ',')
        return s.replace(',0', '') + 'k'
    return int(x)


def plot_histogram(
    data_series: pd.Series,
    ax: Optional[plt.Axes] = None,
    bins: int = 30,
    kde: bool = True,
    title: Optional[str] = None,
    color: str = "#3498db",
    alpha: float = 0.7,
    style: str = 'whitegrid',
    show_mean: bool = False,
    show_median: bool = False,
    show_perc_on_bar: bool = False,
    tick_labelsize: int = 12,
    save_path: Optional[str] = None,
    save_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Plota um histograma completo com formatação de moeda (R$), decimais com vírgula,
    eixos inteligentes e opção de exportação em alta qualidade.
    """
    if not isinstance(data_series, pd.Series):
        raise TypeError("O parâmetro 'data_series' deve ser uma pandas Series.")

    # --- 2. LÓGICA DE FORMATAÇÃO ADICIONADA ---
    try:
        # Tenta usar o locale pt_BR para a formatação ideal
        locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
        FMT_NUM = lambda val: locale.format_string("%.2f", val, grouping=True)
        axis_formatter_x = FuncFormatter(lambda x, p: locale.format_string("%.0f", x, grouping=True))

    except locale.Error:
        # Fallback para formatação manual se o locale pt_BR não estiver disponível
        # print("Aviso: Locale 'pt_BR.UTF-8' não encontrado. Usando formatação manual.")
        FMT_NUM = lambda val: f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        axis_formatter_x = FuncFormatter(lambda x, p: f"{int(x):,}".replace(',', '.'))
    # --- FIM DA LÓGICA DE FORMATAÇÃO ---

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False

    with sns.axes_style(style):
        sns.histplot(data=data_series, bins=bins, kde=kde, ax=ax, color=color, alpha=alpha, edgecolor='white', linewidth=0.5)
        
        # Formatação dos eixos
        ax.yaxis.set_major_formatter(FuncFormatter(human_readable_formatter))
        ax.xaxis.set_major_formatter(axis_formatter_x) # <-- 4. FORMATADOR APLICADO NO EIXO X

        total_displayed_perc = 0.0

        if show_perc_on_bar:
            total = len(data_series)
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    percentage_val = (height / total) * 100
                    total_displayed_perc += percentage_val
                    percentage_str = f'{percentage_val:.1f}'.replace('.', ',') + '%'
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ax.annotate(text=percentage_str, xy=(x, y), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=11, color='dimgray', fontweight='bold')

        final_title = title if title else f'Distribuição de {data_series.name}'
        ax.set_title(final_title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel(data_series.name, fontsize=14, labelpad=15)
        ax.set_ylabel('Frequência', fontsize=14, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

        handles, labels = [], []

        if show_perc_on_bar:
            total_label = f'Total Exibido: {total_displayed_perc:.1f}'.replace('.', ',') + '%'
            handles.append(Rectangle((0,0), 0, 0, fill=False, edgecolor='none', visible=False))
            labels.append(total_label)

        if show_mean:
            mean_val = data_series.mean()
            label = f'Média: {FMT_NUM(mean_val)}' # <-- 3. FORMATADOR APLICADO NA MÉDIA
            ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2.5)
            handles.append(Line2D([0], [0], color='#e74c3c', lw=2.5, linestyle='--'))
            labels.append(label)

        if show_median:
            median_val = data_series.median()
            label = f'Mediana: {FMT_NUM(median_val)}' # <-- 3. FORMATADOR APLICADO NA MEDIANA
            ax.axvline(median_val, color='#9b59b6', linestyle=':', linewidth=2.5)
            handles.append(Line2D([0], [0], color='#9b59b6', lw=2.5, linestyle=':'))
            labels.append(label)

        if handles:
            ax.legend(handles, labels, fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=False, borderpad=1)

        sns.despine(ax=ax)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    if created_fig:
        plt.tight_layout()
    if save_path:
        default_save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.1}
        if save_kwargs:
            default_save_kwargs.update(save_kwargs)
        print(f"Salvando o gráfico em '{save_path}' com os parâmetros: {default_save_kwargs}")
        fig.savefig(save_path, **default_save_kwargs)
    if created_fig:
        plt.show()
        
        
        
        
        
        
        
        
        
    # Importações necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import numpy as np
from matplotlib.ticker import FuncFormatter

# Função auxiliar para formatação inteligente do eixo Y
def human_readable_formatter(x, pos):
    """
    Formata os números do eixo em um formato legível (ex: 400000 -> 400k).
    Usa vírgula como separador decimal quando necessário.
    """
    if x >= 1_000_000:
        s = f'{x/1_000_000:.1f}'.replace('.', ',')
        return s.replace(',0', '') + 'M'
    if x >= 1_000:
        s = f'{x/1_000:.1f}'.replace('.', ',')
        return s.replace(',0', '') + 'k'
    return int(x)


def plot_top_categorical(
    data_series: pd.Series,
    ax: Optional[plt.Axes] = None,
    top_n: int = 7,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = 'Frequência (Contagem)',
    color: str = '#5cb88a',
    save_path: Optional[str] = None,
    save_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Plota um gráfico de barras com as N categorias mais frequentes de uma coluna,
    exibindo a frequência absoluta/relativa e a soma das porcentagens exibidas.
    Os valores percentuais são formatados com vírgula como separador decimal.
    Os rótulos do eixo Y são formatados de forma inteligente (ex: 400k).
    """
    if not isinstance(data_series, pd.Series):
        raise TypeError("O parâmetro 'data_series' deve ser uma pandas Series.")

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=(15, 8))
    else:
        fig = ax.get_figure()

    total_count = len(data_series)
    counts = data_series.value_counts().nlargest(top_n)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color=color, alpha=1.0)
    
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_formatter))

    # Adiciona os valores em cima de cada barra
    for p in ax.patches:
        absolute_freq = int(p.get_height())
        relative_freq_pct = (absolute_freq / total_count) * 100
        
        # --- ALTERAÇÃO AQUI ---
        # Formata o número absoluto com ponto como separador de milhar
        freq_str = f'{absolute_freq:,}'.replace(',', '.')
        
        pct_str = f'{relative_freq_pct:.1f}'.replace('.', ',')
        
        # Usa o número já formatado (freq_str) no rótulo final
        label = f'{freq_str}\n({pct_str}%)'
        
        ax.annotate(label,
                    (p.get_x() + p.get_width() / 2., absolute_freq),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=11, color='#333333', fontweight='bold')

    final_title = title if title else f'Top {top_n} Categorias Mais Frequentes em "{data_series.name}"'
    ax.set_title(final_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel if xlabel else data_series.name, fontsize=12, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    sns.despine(ax=ax)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Bloco para calcular e exibir a soma das porcentagens
    sum_of_top_n_pct = (counts.sum() / total_count) * 100
    total_pct_str = f'{sum_of_top_n_pct:.1f}'.replace('.', ',')
    ax.text(0.98, 0.95,
            f'Total Exibido: {total_pct_str}%',
            transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='#333333',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    if create_fig:
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        default_save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.1}
        if save_kwargs:
            default_save_kwargs.update(save_kwargs)
        print(f"Salvando o gráfico em '{save_path}' com os parâmetros: {default_save_kwargs}")
        fig.savefig(save_path, **default_save_kwargs)

    if create_fig:
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from typing import Optional, Dict, Any

def human_readable_formatter(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.0f}M'
    elif x >= 1_000:
        return f'{x/1_000:.0f}k'
    return f'{x:.0f}'

def plot_categorical_distribution(
    data_series: pd.Series,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = 'Frequência (Contagem)',
    color: str = '#5cb88a',
    sort_by_index: bool = False,
    xtick_rotation: int = 0  # <--- NOVO PARÂMETRO ADICIONADO AQUI
):
    """
    Plota um gráfico de barras com TODAS as categorias de uma coluna,
    exibindo a frequência absoluta e relativa em cada barra.
    
    Permite rotacionar os rótulos do eixo X.
    """
    if not isinstance(data_series, pd.Series):
        raise TypeError("O parâmetro 'data_series' deve ser uma pandas Series.")

    counts = data_series.value_counts()
    
    if sort_by_index:
        counts = counts.sort_index()

    total_count = len(data_series)

    create_fig = ax is None
    if create_fig:
        num_categories = len(counts)
        fig_width = max(15, num_categories * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, 8))
    else:
        fig = ax.get_figure()

    sns.barplot(x=counts.index, y=counts.values, ax=ax, color=color)
    
    ax.yaxis.set_major_formatter(FuncFormatter(human_readable_formatter))

    for p in ax.patches:
        absolute_freq = int(p.get_height())
        relative_freq_pct = (absolute_freq / total_count) * 100
        
        freq_str = f'{absolute_freq:,}'.replace(',', '.')
        pct_str = f'{relative_freq_pct:.1f}'.replace('.', ',')
        label = f'{freq_str}\n({pct_str}%)'
        
        ax.annotate(label,
                    (p.get_x() + p.get_width() / 2., absolute_freq),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=11, color='#333333', fontweight='bold')

    final_title = title if title else f'Distribuição de Frequência em "{data_series.name}"'
    ax.set_title(final_title, fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlabel(xlabel if xlabel else data_series.name, fontsize=12, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    
    # --- ALTERAÇÃO FEITA AQUI ---
    # O valor da rotação agora é definido pelo novo parâmetro
    ax.tick_params(axis='x', rotation=xtick_rotation, labelsize=12)
    
    ax.tick_params(axis='y', labelsize=11)
    sns.despine(ax=ax)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if create_fig:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
























import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

    # --- Função de Avaliação ---
def avaliar_treino(nome_modelo, y_teste, previsao, tempo_treino):
    """
    Avalia o desempenho de um modelo de machine learning e seu tempo de treino.

    Args:
        nome_modelo: O nome do modelo (string).
        y_teste: Os valores reais da variável alvo (array-like).
        previsao: Os valores previstos pelo modelo (array-like).
        tempo_treino: O tempo que o modelo levou para treinar (float, em segundos).

    Returns:
        Uma string formatada com o tempo de treino, R² e RMSE do modelo.
    """

    r2 = r2_score(y_teste, previsao)
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_teste, previsao))
    
    return f'Modelo {nome_modelo}:\nTempo de Treino: {tempo_treino:.2f}s\nR²: {r2:.2%}\nRMSE: {rmse:.2f}'



















import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

    # --- Função de Avaliação ---
def avaliar_teste(nome_modelo, y_teste, previsao):
    """
    Avalia o desempenho de um modelo de machine learning e seu tempo de treino.

    Args:
        nome_modelo: O nome do modelo (string).
        y_teste: Os valores reais da variável alvo (array-like).
        previsao: Os valores previstos pelo modelo (array-like).
        tempo_treino: O tempo que o modelo levou para treinar (float, em segundos).

    Returns:
        Uma string formatada com o tempo de treino, R² e RMSE do modelo.
    """

    r2 = r2_score(y_teste, previsao)
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_teste, previsao))
    
    return f'Modelo {nome_modelo}:\nR²: {r2:.2%}\nRMSE: {rmse:.2f}'



















import pandas as pd

def agrupar_categorias_raras(df, nome_coluna, threshold, novo_nome_categoria='Outros'):
    """
    Agrupa categorias de baixa frequência em uma coluna de um DataFrame.

    Parâmetros:
    -----------
    df : pd.DataFrame
        O DataFrame que contém os dados.
    nome_coluna : str
        O nome da coluna categórica a ser modificada.
    threshold : int
        O número mínimo de ocorrências para uma categoria não ser agrupada.
        Categorias com contagem < threshold serão agrupadas.
    novo_nome_categoria : str, opcional
        O nome para a nova categoria que agrupará as raras (padrão: 'Outros').

    Retorna:
    --------
    pd.DataFrame
        DataFrame com a coluna modificada.
    """
    
    # 1. Contar a frequência de cada categoria na coluna especificada
    counts = df[nome_coluna].value_counts()

    # 3. Identificar as categorias com contagem abaixo do limite (threshold)
    to_group = counts[counts < threshold].index

    # 4. Usar .loc e .isin() para substituir todas as categorias raras de uma só vez
    df.loc[df[nome_coluna].isin(to_group), nome_coluna] = novo_nome_categoria

    return print(df[nome_coluna].value_counts())






































