# lwreq: lmfit, plotly, shinywidgets, scipy, numpy

import numpy as np
import plotly.graph_objects as px
from lmfit import Model
from shiny import App, render, ui
from shinywidgets import output_widget, render_widget

# 1. Definiçao do Modelo Fisicao (Exemplo: Queda Livre/ Movimento Uniformente Variado)
# s = s0 + v0 * t + 0.5 * g * t^2
def modelo_quadratico(t, s0, v0, g):
    return s0 + v0 * t + 0.5 * g * t**2

# Interface do Usuario (UI)
app_ui = ui.page_fluid(
    ui.panel_title("LABFIS - Ajuste de Curvas ODR (Wasm)"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_slider("s0_init", "Posição Inicial (m)", 0.0, 10.0, 0.0),
            ui.input_slider("v0_init", "Velocidade Inicial (m/s)", 0.0, 20.0, 2.0),
            ui.input_slider("g_init", "Aceleração (m/s²)", 5.0, 15.0, 9.8),
            ui.hr(),
            ui.markdown(
                "**Dados Experimentais:**\nModifique os sliders para ajustar a estimativa inicial do LMFIT usando o algorítmo ODR."
            ),
        ),
        ui.card(ui.card_header("Gráfico de Ajuste (Plotly)"), output_widget("plot")),
        ui.card(
            ui.card_header("Resultados do Ajuste Estatístico"),
            ui.output_text_verbatim("resultado_fit"),
        ),
    ),
)

# Logica do Servidor (Server)
def server(input, output, session):
    # Simulando dados experimentais com erros em X, e Y (t e s)
    # Em um cenario real, voce pode substituir por um input de arquivo CSV
    np.random.seed(42)
    t_exp = np.linspace(0, 4, 15)
    # Valor real: s0=0.5, v0=3.0, g=9.81
    s_real = modelo_quadratico(t_exp, 0.3, 3.0, 9.81)
    # Adicionando ruído
    t_err = 0.1 * np.ones_like(t_exp)
    s_err = 0.5 * np.ones_like(t_exp)
    t_exp += np.random.normal(0, 0.05, size=len(t_exp))
    s_exp = s_real + np.random.normal(0, 0.4, size=len(s_real))

    @render_widget
    def plot():
        # Executando o ajuste via LMFIT utilizando o método ODR (Orthogonal Distance Regression)
        # O ORD é crucial na física pois considera incertezas em X e em Y
        model = Model(modelo_quadratico)
        params = model.make_params(
            s0=input.s0_init(), v0=input.v0_init(), g=input.g_init()
        )

        # Ajuste ortogonal considerando ambos os erros
        result = model.fit(
            s_exp, params, t=t_exp, xerr=t_err, weights=1.0 / s_err, method="odr"
        )
        
        # Gerando linha de ajuste
        t_fit = np.linspace(0, 4, 100)
        s_fit = result.eval(t=t_fit)

        # Criando o grafico interativo com plotly.js no bacjkend Wasm
        fig = px.Figure()

        # Pontos experimentais com barras de erros nos dois eixos
        fig.add_trace(
            px.Scatter(
                x=t_exp,
                y=s_exp,
                mode="markers",
                name="Dados Expeerimentais",
                error_x=dict(type="data", array=t_err, visible=True),
                error_y=dict(type="data", array=s_err, visible=True),
                marker=dict(color="orange", size=8),
            )
        )

        # Linha de ajuste teórico
        fig.add_trace(
            px.Scatter(
                x=t_fit, y=s_fit, mode="lines", name="Ajuste ODR", line=dict(color="teal")
            )
        )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Tempo (s)",
            yaxis_title="Posição (m)",
            template="plotly_white",
        )
        return fig
    @render.text
    def resultado_fit():
        # Recalcula o fit para exibir o report textual na tela
        model = Model(modelo_quadratico)
        params = model.make_params(
            s0=input.s0_init(), v0=input.v0_init(), g=input.g_init()
        )
        result = model.fit(
            s_exp, params, t=t_exp, xerr=t_err, weights=1.0 / s_err, method="odr"
        )

        # Retorna o relatorio completo do LMFIT (Chi-quadrado, R², incertezas dos parametros)
        return result.fit_report()
    
app = App(app_ui, server) 
