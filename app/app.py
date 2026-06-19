import asyncio
from shiny import App, render, ui
from shinywidgets import output_widget, render_widget

# 1. Função assíncrona para forçar a instalação do lmfit dentro do navegador
async def inicializar_ambiente():
    import micropip
    # Instala o lmfit e garante que as dependências (scipy, numpy) venham juntas
    await micropip.install("lmfit")
    await micropip.install("plotly")

# Executa a instalação em background no carregamento do WebAssembly
asyncio.run(inicializar_ambiente())

# 2. Agora que temos a garantia da instalação, importamos o motor matemático
import numpy as np
import plotly.graph_objects as px
from lmfit import Model

# 3. Definição do Modelo Físico
def modelo_quadratico(t, s0, v0, g):
    return s0 + v0 * t + 0.5 * g * t**2

# Interface do Usuário (UI)
app_ui = ui.page_fluid(
    ui.panel_title("Laboratório de Física - Ajuste de Curvas ODR (Wasm)"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_slider("s0_init", "S0 Inicial (m)", 0.0, 10.0, 0.0),
            ui.input_slider("v0_init", "V0 Inicial (m/s)", 0.0, 20.0, 2.0),
            ui.input_slider("g_init", "g Inicial (m/s²)", 5.0, 15.0, 9.8),
            ui.hr(),
            ui.markdown(
                "**Dados Experimentais:**\nModifique os sliders para ajustar a estimativa inicial do LMFIT usando o algoritmo ODR."
            ),
        ),
        ui.card(ui.card_header("Gráfico de Ajuste (Plotly)"), output_widget("plot")),
        ui.card(
            ui.card_header("Resultados do Ajuste Estatístico"),
            ui.output_text_verbatim("resultado_fit"),
        ),
    ),
)

# Lógica do Servidor (Server)
def server(input, output, session):
    np.random.seed(42)
    t_exp = np.linspace(0, 4, 15)
    s_real = modelo_quadratico(t_exp, 0.5, 3.0, 9.81)
    t_err = 0.1 * np.ones_like(t_exp)
    s_err = 0.5 * np.ones_like(t_exp)
    t_exp += np.random.normal(0, 0.05, size=len(t_exp))
    s_exp = s_real + np.random.normal(0, 0.4, size=len(s_real))

    @render_widget
    def plot():
        model = Model(modelo_quadratico)
        params = model.make_params(
            s0=input.s0_init(), v0=input.v0_init(), g=input.g_init()
        )

        result = model.fit(
            s_exp, params, t=t_exp, xerr=t_err, weights=1.0 / s_err, method="odr"
        )

        t_fit = np.linspace(0, 4, 100)
        s_fit = result.eval(t=t_fit)

        fig = px.Figure()
        fig.add_trace(
            px.Scatter(
                x=t_exp,
                y=s_exp,
                mode="markers",
                name="Dados (Exp)",
                error_x=dict(type="data", array=t_err, visible=True),
                error_y=dict(type="data", array=s_err, visible=True),
                marker=dict(color="orange", size=8),
            )
        )

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
        model = Model(modelo_quadratico)
        params = model.make_params(
            s0=input.s0_init(), v0=input.v0_init(), g=input.g_init()
        )
        result = model.fit(
            s_exp, params, t=t_exp, xerr=t_err, weights=1.0 / s_err, method="odr"
        )
        return result.fit_report()

app = App(app_ui, server)