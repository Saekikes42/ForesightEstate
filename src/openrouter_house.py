"""OpenRouter text-to-image integration for property previews."""
import base64
import requests

MODEL_ID = "inclusionai/ming-image-0.1-design"
MODEL_LABEL = "Ming Image 0.1 Design (бесплатно)"

def build_house_prompt(row, estimated_price=None):
    address_parts = str(row.get("address", "")).split("/")
    location = ", ".join(part.strip() for part in address_parts if part and part.strip())
    subtype = str(row.get("sub_type") or "недвижимость")
    subtype_ru = {
        "Daire": "квартира", "Rezidans": "квартира в резиденции",
        "Villa": "вилла", "Müstakil Ev": "отдельно стоящий дом",
        "Yazlık": "загородный дом", "Çiftlik Evi": "дом на ферме",
        "Köşk": "особняк", "Yalı": "дом у воды",
    }.get(subtype, subtype)
    room_count = str(row.get("room_count", "не указано"))
    bedroom_count, _, living_room_count = room_count.partition("+")
    area = float(row.get("size", 0))
    age = row.get("building_age", "не указан")
    building_floors = row.get("total_floor_count", "не указано")
    floor = row.get("floor_no", "не указан")
    heating = row.get("heating_type") or "не указано"
    estimate = f" Расчётная стоимость: {estimated_price:,.0f} TRY.".replace(",", " ") if estimated_price is not None else ""
    return (
        "Фотореалистичная архитектурная фотография жилой недвижимости для приложения оценки стоимости. "
        f"Объект: {subtype_ru}. Местоположение: {location or 'Турция'}. "
        f"Общая площадь: {area:g} м². Планировка: {bedroom_count} спальни и {living_room_count} гостиная(ые). "
        f"Возраст здания: {age} лет. Этажность здания: {building_floors}; объект расположен на этаже {floor}. "
        f"Тип отопления: {heating}." + estimate + " "
        "Покажи правдоподобный внешний вид и масштаб дома с подходящим фасадом и окружением. "
        "Естественный архитектурный фотостиль, реалистичные материалы и освещение. "
        "Без людей, текста, подписей, логотипов и водяных знаков."
    )

def generate_house_image(row, api_key, estimated_price=None):
    if not api_key:
        raise ValueError("Для генерации фото задайте OPENROUTER_API_KEY в переменных окружения или secrets.toml.")
    response = requests.post(
        "https://openrouter.ai/api/v1/images",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": MODEL_ID, "prompt": build_house_prompt(row, estimated_price), "n": 1},
        timeout=180,
    )
    if response.status_code == 429:
        raise RuntimeError("Бесплатный лимит генерации временно исчерпан. Повторите позже.")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json().get("error", {}).get("message", response.text[:400])
        except ValueError:
            detail = response.text[:400]
        raise RuntimeError(f"OpenRouter вернул ошибку: {detail}") from exc
    try:
        image = response.json()["data"][0]
        return base64.b64decode(image["b64_json"]), image.get("media_type", "image/png")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenRouter не вернул файл изображения.") from exc
