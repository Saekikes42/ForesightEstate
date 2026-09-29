"""ForesightEstate: valuation, analytics, house concepts and help."""
import base64
import os
from datetime import date
import pandas as pd
import unicodedata
import plotly.express as px
import streamlit as st
from src.app_model import ROOT, METRICS, MODELS, load_model, predict
from src.openrouter_house import generate_house_image, MODEL_LABEL

st.set_page_config(page_title="ForesightEstate", page_icon=str(ROOT / "assets/avatar.png"), layout="wide")
cache = getattr(st, "cache_resource", None) or st.experimental_singleton

@cache
def models():
    return {name: load_model(name) for name in MODELS}

@cache
def catalog():
    cols = ["address", "sub_type", "heating_type", "listing_type", "price_currency", "price", "size"]
    df = pd.read_csv(ROOT / "data/real_estate_data.csv", usecols=cols, low_memory=False)
    df = df.loc[df.listing_type.eq(1) & df.price_currency.eq("TRY")].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.loc[df.price.gt(0)]
    split = df.address.fillna("").str.split("/", n=2, expand=True)
    df["city"], df["district"], df["neighborhood"] = split[0], split[1], split[2]
    return df

banner = base64.b64encode((ROOT / "assets/hero-house-v2.webp").read_bytes()).decode("ascii")
styles = (ROOT / "assets/app.css").read_text(encoding="utf-8").replace("__HERO__", banner)
st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><div class="hero-inner">'
    '<div class="hero-kicker">FORESIGHTESTATE / ИССЛЕДОВАНИЕ НЕДВИЖИМОСТИ</div>'
    '<div class="hero-title">Ваш дом.<br>Ваша цена.</div>'
    '<div class="hero-subtitle">Оцените стоимость жилья, изучите рынок и представьте будущий дом. '
    'Прозрачные модели на данных объявлений Турции.</div>'
    '<span class="hero-tag">Оценка стоимости в турецких лирах</span>'
    '</div></div>', unsafe_allow_html=True,
)

tabs = st.tabs(["Прогноз стоимости", "Аналитика рынка", "Справка"])

with tabs[0]:
    st.markdown('<div class="section-eyebrow">01 / Прогноз</div>', unsafe_allow_html=True)
    st.header("Оцените недвижимость")
    st.markdown('<div class="section-copy">Заполните характеристики объекта. Обязательные поля отмечены звёздочкой.</div>', unsafe_allow_html=True)
    try:
        df = catalog()
        model_bank = models()
    except Exception as exc:
        st.error(f"Не удалось загрузить данные или модель: {exc}")
        st.stop()

    st.markdown('<div class="group-title"><span>01</span> Модель и расположение</div>', unsafe_allow_html=True)
    model_col, _ = st.columns([1, 2])
    with model_col:
        model_name = st.selectbox("Модель оценки *", list(MODELS))
    a, b, c = st.columns(3)
    with a:
        city = st.selectbox("Провинция *", ["Выберите провинцию"] + sorted(x for x in df.city.dropna().unique() if x))
    city_df = df.loc[df.city.eq(city)]
    with b:
        district = st.selectbox("Район *", ["Выберите район"] + sorted(x for x in city_df.district.dropna().unique() if x))
    with c:
        neighborhood = st.selectbox(
            "Микрорайон",
            ["Не указан"] + sorted(x for x in city_df.loc[city_df.district.eq(district), "neighborhood"].dropna().unique() if x),
        )
    if model_name == "Средний сегмент":
        st.markdown('<div class="info-card">Модель среднего сегмента обучена на объектах с ценой <strong>138 000–750 000 TRY</strong>. '
                    'Результат за пределами этого диапазона может быть ненадёжным.</div>', unsafe_allow_html=True)

    st.markdown('<div class="group-title"><span>02</span> Характеристики объекта</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        sub_type = st.selectbox("Тип недвижимости *", sorted(df.sub_type.dropna().unique()))
        area = st.number_input("Площадь, м² *", min_value=15.0, max_value=100000.0, value=100.0, step=5.0)
        heating = st.selectbox("Отопление", ["Не указано"] + sorted(df.heating_type.dropna().unique()))
    with b:
        bedrooms = st.number_input("Спальни *", min_value=0, max_value=20, value=2, step=1)
        living_rooms = st.number_input("Гостиные *", min_value=0, max_value=10, value=1, step=1)
        year = st.number_input("Год постройки *", min_value=1800, max_value=date.today().year, value=2015, step=1)
    with c:
        building_floors = st.number_input("Этажей в здании *", min_value=1, max_value=100, value=5, step=1)
        floor_no = st.number_input("Этаж объекта *", min_value=-5, max_value=100, value=2, step=1)
        st.markdown('<div class="info-card"><strong>Как считаем:</strong> вводные параметры преобразуются так же, как при обучении моделей.</div>', unsafe_allow_html=True)
    st.markdown('<div class="detail-strip"><span class="detail-pill">Продажа в TRY</span>'
                '<span class="detail-pill">Две модели на выбор</span>'
                '<span class="detail-pill">Диапазон с учётом MAE</span></div>', unsafe_allow_html=True)
    address = f"{city}/{district}" + (f"/{neighborhood}" if neighborhood != "Не указан" else "")
    row = {
        "address": address, "sub_type": sub_type,
        "heating_type": None if heating == "Не указано" else heating,
        "size": area, "room_count": f"{bedrooms}+{living_rooms}",
        "building_age": str(date.today().year - year),
        "total_floor_count": str(building_floors), "floor_no": str(floor_no),
    }
    if st.button("Рассчитать стоимость"):
        errors = []
        if city == "Выберите провинцию" or district == "Выберите район":
            errors.append("Выберите провинцию и район.")
        if bedrooms + living_rooms == 0:
            errors.append("Укажите хотя бы одну комнату.")
        if floor_no > building_floors:
            errors.append("Этаж объекта не может превышать число этажей здания.")
        if sub_type in ("Daire", "Rezidans") and area > 3000:
            errors.append("Для квартиры или резиденции площадь должна быть не более 3000 м².")
        for error in errors:
            st.error(error)
        if not errors:
            try:
                estimate = float(predict(model_bank[model_name], row))
                mae, r2 = METRICS[model_name]
                st.session_state["valuation_result"] = {
                    "model": model_name, "row": row, "value": estimate,
                    "mae": mae, "r2": r2, "photo": None, "photo_type": None,
                }
            except Exception as exc:
                st.error(f"Не удалось рассчитать прогноз: {exc}")

    quote = st.session_state.get("valuation_result")
    same_inputs = bool(quote and quote["model"] == model_name and quote["row"] == row)
    if quote and not same_inputs:
        st.caption("Характеристики изменились. Пересчитайте стоимость, чтобы получить фото для новых параметров.")
    if same_inputs:
        number = lambda n: f"{n:,.0f}".replace(",", " ")
        st.markdown(
            f'<div class="result-card"><div class="result-kicker">Оценка модели</div>'
            f'<div class="result-price">{number(quote["value"])} TRY</div>'
            f'<div class="result-note">Ориентир с учётом MAE: {number(max(0, quote["value"]-quote["mae"]))}–{number(quote["value"]+quote["mae"])} TRY</div>'
            '</div>', unsafe_allow_html=True,
        )
        st.caption(f'Историческая MAE: {number(quote["mae"])} TRY; R²: {quote["r2"]:.3f}. '
                   'Диапазон ±MAE не является калиброванным доверительным интервалом.')
        st.info("Прогноз основан на исторических объявлениях, а не на текущих рыночных ценах.")
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
        except Exception:
            api_key = os.getenv("OPENROUTER_API_KEY", "")
        st.caption("Генерация: " + MODEL_LABEL)
        if st.button("Сгенерировать фото объекта"):
            try:
                with st.spinner("Создаём фото дома по характеристикам объекта..."):
                    photo, media_type = generate_house_image(quote["row"], api_key, quote["value"])
                    st.session_state["valuation_result"]["photo"] = photo
                    st.session_state["valuation_result"]["photo_type"] = media_type
            except Exception as exc:
                st.error(f"Не удалось создать фото: {exc}")
        quote = st.session_state.get("valuation_result")
        if quote and quote.get("photo"):
            st.image(quote["photo"], caption="Сгенерированный визуальный образ объекта", use_column_width=True)
        if not api_key:
            st.caption("Для работы кнопки задайте OPENROUTER_API_KEY в переменных окружения или в .streamlit/secrets.toml.")

with tabs[1]:
    st.markdown('<div class="section-eyebrow">02 / Аналитика</div>', unsafe_allow_html=True)
    st.header("Исследуйте рынок")
    st.markdown('<div class="section-copy">Цены продаж в TRY по типам недвижимости и провинциям. Выберите область и тип для фильтрации графиков.</div>', unsafe_allow_html=True)
    city_options = ["Все провинции"] + sorted(x for x in df.city.dropna().unique() if x)
    type_options = ["Все типы"] + sorted(x for x in df.sub_type.dropna().unique() if x)
    f1, f2 = st.columns(2)
    with f1:
        selected_city = st.selectbox("Провинция для анализа", city_options)
    with f2:
        selected_type = st.selectbox("Тип недвижимости для анализа", type_options)
    market = df
    if selected_city != "Все провинции":
        market = market.loc[market.city.eq(selected_city)]
    if selected_type != "Все типы":
        market = market.loc[market.sub_type.eq(selected_type)]
    if market.empty:
        st.info("По этому сочетанию фильтров объявлений нет.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Объявлений", f"{len(market):,}".replace(",", " "))
        k2.metric("Медианная цена", f"{market.price.median():,.0f} TRY".replace(",", " "))
        k3.metric("Средняя цена", f"{market.price.mean():,.0f} TRY".replace(",", " "))
        p99 = market.price.quantile(.99)
        col1, col2 = st.columns(2)
        with col1:
            fig_price = px.histogram(market.loc[market.price.le(p99)], x="price", nbins=45,
                                     title="Распределение цен (до 99-го процентиля)",
                                     labels={"price": "Цена, TRY", "count": "Объявления"},
                                     color_discrete_sequence=["#A9784F"])
            st.plotly_chart(fig_price, use_container_width=True)
        with col2:
            by_type = market.groupby("sub_type").price.agg(median="median", count="size").nlargest(12, "count").reset_index()
            fig_type = px.bar(by_type, x="sub_type", y="median", hover_data=["count"],
                              title="Медианная цена по типу объекта",
                              labels={"sub_type": "Тип", "median": "Медианная цена, TRY", "count": "Объявления"},
                              color_discrete_sequence=["#A9784F"])
            st.plotly_chart(fig_type, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            by_city = market.groupby("city").price.agg(median="median", count="size").nlargest(15, "count").reset_index()
            fig_city = px.bar(by_city, x="city", y="median", hover_data=["count"],
                              title="Медианная цена по провинциям",
                              labels={"city": "Провинция", "median": "Медианная цена, TRY", "count": "Объявления"},
                              color_discrete_sequence=["#384B55"])
            st.plotly_chart(fig_city, use_container_width=True)
        with col2:
            scatter_source = market.loc[market["size"].between(15, 1000) & market.price.le(p99)]
            scatter = scatter_source.sample(n=min(2500, len(scatter_source)), random_state=42) if len(scatter_source) else scatter_source
            if not scatter.empty:
                fig_area = px.scatter(scatter, x="size", y="price", color="sub_type",
                                      title="Площадь и цена (до 2 500 объявлений)",
                                      labels={"size": "Площадь, м²", "price": "Цена, TRY", "sub_type": "Тип"},
                                      opacity=.55)
                st.plotly_chart(fig_area, use_container_width=True)
        st.subheader("Карта объявлений по провинциям")
        st.caption("Точки стоят в центрах провинций: исходный CSV не содержит координат объявлений.")
        geo_path = ROOT / "data" / "external" / "turkey_provinces.csv"
        geo = pd.read_csv(geo_path, usecols=["province_name", "latitude", "longitude"])
        normalize = lambda value: unicodedata.normalize("NFKD", str(value).replace("ı", "i").replace("İ", "I")).encode("ascii", "ignore").decode().lower().strip()
        geo["city_key"] = geo.province_name.map(normalize)
        points = market.groupby("city").price.agg(median="median", count="size").reset_index()
        points["city_key"] = points.city.map(normalize)
        points = points.merge(geo, on="city_key", how="inner")
        if not points.empty:
            map_fig = px.scatter_map(
                points, lat="latitude", lon="longitude", size="count", color="median",
                hover_name="city", hover_data={"count": True, "median": ":,.0f", "latitude": False, "longitude": False, "city_key": False},
                zoom=4.2, center={"lat": 39, "lon": 35}, map_style="open-street-map",
                color_continuous_scale=["#E7D6BF", "#A9784F", "#26343D"],
                labels={"count": "Объявления", "median": "Медианная цена, TRY"},
            )
            st.plotly_chart(map_fig, use_container_width=True)
        st.caption("Координаты центров провинций: turkiye-provinces-dataset, CC BY 4.0.")

    st.subheader("Качество моделей")
    compare = pd.DataFrame({
        "Модель": ["Общая", "Средний сегмент"],
        "MAE, TRY": [196033, 62140],
        "R²": [0.357, 0.605],
    })
    st.dataframe(compare, use_container_width=True)
    st.caption("Метрики получены на разных выборках: общая модель — validation, средний сегмент — test. Сравнивайте их с учётом этого различия.")
with tabs[2]:
    st.markdown('<div class="section-eyebrow">03 / Справка</div>', unsafe_allow_html=True)
    st.header("Как пользоваться")
    st.markdown('<div class="info-card"><strong>Прогноз.</strong> Выберите модель, провинцию и район, укажите тип объекта, площадь в м², комнаты, год постройки и этажи. Нажмите «Рассчитать стоимость».</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><strong>Точность.</strong> Общая модель охватывает широкий набор объектов. Средняя модель обучена на ценах 138–750 тыс. TRY. Метрики посчитаны на разных выборках и не означают, что одна модель лучше другой для любого дома.</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><strong>Ограничения.</strong> В данных нет точных координат, состояния ремонта, вида из окна и актуальной рыночной динамики. ±MAE — ориентир исторической ошибки, а не гарантия. Фото объекта можно сгенерировать после оценки.</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><strong>Карта.</strong> Координаты центров провинций: brkunver/turkiye-provinces-dataset, CC BY 4.0. Маркеры не являются адресами объектов.</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><strong>Проект.</strong> Команда ForesightEstate. Контакт — репозиторий GitHub Saekikes42/ForesightEstate. Версия приложения 1.2.</div>', unsafe_allow_html=True)
