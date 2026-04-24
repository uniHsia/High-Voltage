import json
import random
import string
import hashlib

from django.conf import settings
from django.contrib.auth.hashers import make_password, check_password


def ensure_users_file():
    path = settings.USERS_JSON_PATH
    if not path.exists():
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)


def load_users():
    ensure_users_file()
    with open(settings.USERS_JSON_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def save_users(users):
    with open(settings.USERS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=4)


def hash_password(password: str) -> str:
    return make_password(password)


def user_exists(username: str) -> bool:
    users = load_users()
    return any(u['username'] == username for u in users)


def register_user(username: str, email: str, password: str):
    users = load_users()
    users.append({
        'username': username,
        'email': email,
        'password': hash_password(password)
    })
    save_users(users)


def verify_user(username: str, password: str) -> bool:
    users = load_users()
    legacy_password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    for user in users:
        if user['username'] != username:
            continue

        stored_password = user.get('password', '')
        # 兼容旧版 SHA256 存储；登录成功后自动升级为 Django 安全哈希。
        if stored_password == legacy_password_hash:
            user['password'] = hash_password(password)
            save_users(users)
            return True

        if check_password(password, stored_password):
            return True
    return False


def generate_captcha_text(length=4):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))


def generate_captcha_svg(captcha_text: str) -> str:
    """
    生成 SVG 验证码，不依赖 Pillow 和本地字体。
    """
    width = 130
    height = 48

    bg_colors = ["#f7fbff", "#eef6ff", "#f4faff"]
    line_colors = ["#9cc7f5", "#7fb3ea", "#b7d6f7"]
    text_colors = ["#245b96", "#2f6dac", "#1f4f82"]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" rx="10" ry="10" fill="{random.choice(bg_colors)}"/>'
    ]

    # 干扰线
    for _ in range(5):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        svg_parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{random.choice(line_colors)}" stroke-width="1"/>'
        )

    # 干扰点
    for _ in range(20):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        r = random.randint(1, 2)
        svg_parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{random.choice(line_colors)}" opacity="0.7"/>'
        )

    # 文字
    start_x = 16
    for i, ch in enumerate(captcha_text):
        x = start_x + i * 24
        y = random.randint(30, 36)
        rotate = random.randint(-12, 12)
        color = random.choice(text_colors)
        svg_parts.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-size="26" font-weight="700" '
            f'font-family="Arial, Helvetica, sans-serif" transform="rotate({rotate} {x} {y})">{ch}</text>'
        )

    svg_parts.append('</svg>')
    return ''.join(svg_parts)
