"""Interactive Dash analytics for sale listings in Turkish lira."""
from pathlib import Path
import unicodedata
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

ROOT = Path(__file__).resolve().parent
PALETTE = ["#a9784f", "#384b55", "#a8b4aa", "#dfc5a3"]

def normalized(value):
    return unicodedata.normalize("NFKD", str(value).replace("ı", "i").replace("İ", "I")).encode("ascii", "ignore").decode().lower().strip()

def load_data():
    cols = ["address", "sub_type", "listing_type", "price_currency", "price", "size"]
    frame = pd.read_csv(ROOT / "data/real_estate_data.csv", usecols=cols, low_memory=False)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.loc[(frame.listing_type == 1) & frame.price_currency.eq("TRY") & frame.price.gt(0)].copy()
    frame["city"] = frame.address.fillna("").str.split("/").str[0]
    return frame

DATA = load_data()
GEO = pd.read_csv(ROOT / "data/external/turkey_provinces.csv", usecols=["province_name", "latitude", "longitude"])
GEO["city_key"] = GEO.province_name.map(normalized)
app = Dash(__name__)
app.title = "ForesightEstate — аналитика"
CARD = {"background": "#ffffff", "padding": "22px", "borderRadius": "18px", "boxShadow": "0 8px 30px #15232a10"}
app.layout = html.Div([
    html.Div([
        html.Div("FORESIGHTESTATE / АНАЛИТИКА", style={"fontSize": "12px", "letterSpacing": "3px", "color": "#d7b38b"}),
        html.H1("Рынок недвижимости Турции", style={"fontFamily": "Georgia", "fontSize": "42px"}),
        html.P("Интерактивное исследование объявлений о продаже в TRY. Цены относятся к периоду сбора исходного набора данных."),
    ], style={"background": "#26343d", "color": "#fff", "padding": "42px 6%"}),
    html.Div([
        html.Div([
            html.Label("Провинция"), dcc.Dropdown(id="city", options=[{"label": "Все провинции", "value": "all"}] + [{"label": c, "value": c} for c in sorted(DATA.city.dropna().unique())], value="all", clearable=False),
            html.Label("Тип недвижимости", style={"marginTop": "16px", "display": "block"}), dcc.Dropdown(id="kind", options=[{"label": "Все типы", "value": "all"}] + [{"label": c, "value": c} for c in sorted(DATA.sub_type.dropna().unique())], value="all", clearable=False),
        ], style={**CARD, "marginBottom": "20px"}),
        html.Div(id="stats", style={"marginBottom": "18px", "fontSize": "19px", "fontWeight": "600"}),
        html.Div([dcc.Graph(id="price-hist"), dcc.Graph(id="types-chart")], style={**CARD, "display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(380px,1fr))", "gap": "12px"}),
        html.Div([dcc.Graph(id="city-chart"), dcc.Graph(id="area-chart")], style={**CARD, "display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(380px,1fr))", "gap": "12px", "marginTop": "20px"}),
        html.Div([dcc.Graph(id="map")], style={**CARD, "marginTop": "20px"}),
        html.P("Карта: координаты центров провинций из открытого справочника brkunver/turkiye-provinces-dataset (CC BY 4.0). В исходном CSV координат объявлений нет; точка показывает агрегат провинции, а не адрес объекта.", style={"color": "#59666b"}),
        html.H2("Сравнение моделей", style={"fontFamily": "Georgia"}),
        html.Div([dcc.Graph(id="model-chart")], style=CARD),
        html.P("Метрики получены на разных выборках: общая модель — validation, средний сегмент — test с ценой 138–750 тыс. TRY. Их нельзя считать прямым соревнованием на одной и той же задаче."),
    ], style={"maxWidth": "1440px", "margin": "auto", "padding": "28px 5%"}),
], style={"background": "#f5f1eb", "fontFamily": "Arial, sans-serif", "color": "#26343d", "minHeight": "100vh"})

@app.callback(
    Output("stats", "children"), Output("price-hist", "figure"), Output("types-chart", "figure"),
    Output("city-chart", "figure"), Output("area-chart", "figure"), Output("map", "figure"),
    Output("model-chart", "figure"), Input("city", "value"), Input("kind", "value")
)
def update(city, kind):
    frame = DATA
    if city != "all":
        frame = frame.loc[frame.city.eq(city)]
    if kind != "all":
        frame = frame.loc[frame.sub_type.eq(kind)]
    if frame.empty:
        empty = go.Figure()
        empty.add_annotation(text="Нет объявлений для выбранных фильтров", showarrow=False)
        return "Нет данных", empty, empty, empty, empty, empty, empty
    stats = f"{len(frame):,} объявлений · медиана {frame.price.median():,.0f} TRY · средняя {frame.price.mean():,.0f} TRY".replace(",", " ")
    upper = frame.price.quantile(.99)
    hist = px.histogram(frame.loc[frame.price.le(upper)], x="price", nbins=45, title="Распределение цен (до P99)", color_discrete_sequence=PALETTE)
    typ = frame.groupby("sub_type").price.agg(["median", "size"]).nlargest(12, "size").reset_index()
    types = px.bar(typ, x="sub_type", y="median", hover_data=["size"], title="Медианная цена по типу", color_discrete_sequence=PALETTE)
    cities = frame.groupby("city").price.agg(["median", "size"]).nlargest(15, "size").reset_index()
    city_fig = px.bar(cities, x="city", y="median", hover_data=["size"], title="Медианная цена по провинции", color_discrete_sequence=PALETTE)
    sample = frame.loc[frame["size"].between(15, 1000) & frame.price.le(upper)].sample(n=min(3000, len(frame.loc[frame["size"].between(15, 1000) & frame.price.le(upper)])), random_state=42)
    area = px.scatter(sample, x="size", y="price", color="sub_type", title="Площадь и цена: выборка до 3000 объектов")
    points = frame.groupby("city").price.agg(["median", "size"]).reset_index()
    points["city_key"] = points.city.map(normalized)
    points = points.merge(GEO, on="city_key", how="inner")
    map_fig = px.scatter_map(points, lat="latitude", lon="longitude", size="size", color="median",
                             hover_name="city", hover_data={"size": True, "median": ":,.0f", "latitude": False, "longitude": False},
                             zoom=4.2, center={"lat": 39, "lon": 35}, map_style="open-street-map",
                             title="Провинции: размер — число объявлений, цвет — медианная цена",
                             color_continuous_scale=["#e7d6bf", "#a9784f", "#26343d"])
    model = pd.DataFrame({"Модель": ["Общая", "Средний сегмент"], "MAE, TRY": [196033, 62140], "R²": [.357, .605]})
    model_fig = px.bar(model, x="Модель", y="MAE, TRY", color="Модель", title="Средняя абсолютная ошибка (разные тестовые области)", color_discrete_sequence=PALETTE)
    for figure in (hist, types, city_fig, area, map_fig, model_fig):
        figure.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff", font_color="#26343d", margin=dict(l=20, r=20, t=60, b=25))
    return stats, hist, types, city_fig, area, map_fig, model_fig

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
