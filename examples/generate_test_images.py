"""生成测试图片 — 纯 Python 生成，无需联网。"""
from PIL import Image, ImageDraw, ImageFont
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(OUTPUT, exist_ok=True)


def make_image(filename, color, shapes, text):
    img = Image.new("RGB", (400, 300), color)
    draw = ImageDraw.Draw(img)
    for shape in shapes:
        if shape["type"] == "circle":
            draw.ellipse(shape["coords"], fill=shape["fill"], outline="black")
        elif shape["type"] == "rect":
            draw.rectangle(shape["coords"], fill=shape["fill"], outline="black")
        elif shape["type"] == "line":
            draw.line(shape["coords"], fill=shape["fill"], width=shape.get("width", 3))
    # 文字
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((400 - tw) // 2, 260), text, fill="white", font=font, stroke_width=2, stroke_color="black")
    path = os.path.join(OUTPUT, filename)
    img.save(path)
    print(f"  ✓ {filename}")


# ── 生成 6 张各有特色的测试图 ──
print("生成测试图片...")

make_image("cat_sunset.jpg",         (255, 140, 50),   # 暖橙夕阳色
    [{"type":"circle","coords":[160,80,240,160],"fill":"#FF8C00"}],   # 太阳
    "猫在窗台看夕阳")

make_image("beach.jpg",              (135, 206, 235),  # 天蓝色
    [{"type":"rect","coords":[0,200,400,300],"fill":"#F4A460"},       # 沙滩
     {"type":"rect","coords":[0,0,400,180],"fill":"#87CEEB"}],        # 天空
    "海滩和棕榈树")

make_image("dog_park.jpg",           (144, 238, 144),  # 草绿色
    [{"type":"circle","coords":[280,100,330,150],"fill":"#FF4500"},   # 红球
     {"type":"rect","coords":[0,220,400,300],"fill":"#228B22"}],      # 草地
    "公园里玩球的狗")

make_image("birthday_cake.jpg",      (255, 182, 193),  # 粉红色
    [{"type":"rect","coords":[150,120,250,240],"fill":"#FFFAF0"},     # 蛋糕
     {"type":"circle","coords":[195,80,205,90],"fill":"#FFD700"}],    # 蜡烛
    "生日蛋糕和派对")

make_image("snow_mountain.jpg",      (200, 220, 255),  # 淡蓝色背景
    [{"type":"line","coords":[0,280,200,100,400,280],"fill":"#FFFFFF","width":40},  # 雪山
     {"type":"rect","coords":[0,240,400,300],"fill":"#F0F0F0"}],      # 雪地
    "雪山和森林小屋")

make_image("city_night.jpg",         (25, 25, 80),     # 深蓝色夜空
    [{"type":"rect","coords":[30,150,80,300],"fill":"#FFD700"},       # 高楼亮灯
     {"type":"rect","coords":[150,120,200,300],"fill":"#FF6347"},
     {"type":"rect","coords":[280,170,340,300],"fill":"#00CED1"},
     {"type":"circle","coords":[350,30,370,50],"fill":"#FFFF00"}],    # 月亮
    "城市夜景")

print(f"\n✅ 生成完毕！共 6 张，在 {OUTPUT}")
