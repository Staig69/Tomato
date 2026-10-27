import os
import random
import shutil
import cv2
from ultralytics import SAM

# ==================== НАСТРОЙКА ПУТЕЙ ====================
SOURCE_DIR = "C:/tomato_dataset"

# Куда скрипт создаст правильную структуру проекта
TARGET_BASE_DIR = "../data"
# =========================================================


def split_and_copy_data(val_ratio=0.15, test_ratio=0.15):
    """Шаг 1: Автоматическое разбиение данных на train, val и test"""
    src_images_dir = os.path.join(SOURCE_DIR, "images")
    src_labels_dir = os.path.join(SOURCE_DIR, "labels")

    if not os.path.exists(src_images_dir) or not os.path.exists(src_labels_dir):
        print("Папки не найдены.")
        return False

    # Находим все файлы аннотаций
    all_labels = [f for f in os.listdir(src_labels_dir) if f.endswith('.txt')]
    random.seed(42)  # Фиксируем сид, чтобы разбиение было воспроизводимым
    random.shuffle(all_labels)

    total_count = len(all_labels)
    val_count = int(total_count * val_ratio)
    test_count = int(total_count * test_ratio)

    # Распределяем файлы по независимым множествам
    val_files = set(all_labels[:val_count])
    test_files = set(all_labels[val_count:val_count + test_count])

    print(f"Всего найдено объектов: {total_count}")
    print(f"Распределение: {total_count - val_count - test_count} в train, {val_count} в val, {test_count} в test.")

    for label_file in all_labels:
        # Определяем, в какую выборку отправить файл
        if label_file in val_files:
            split = "val"
        elif label_file in test_files:
            split = "test"
        else:
            split = "train"

        # Определяем имя картинки
        img_file = label_file.replace(".txt", ".jpg")
        if not os.path.exists(os.path.join(src_images_dir, img_file)):
            img_file = label_file.replace(".txt", ".png")

        # Пути назначения
        dst_img_dir = os.path.join(TARGET_BASE_DIR, split, "images")
        dst_box_dir = os.path.join(TARGET_BASE_DIR, split, "labels_box")

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_box_dir, exist_ok=True)

        # Копируем картинку и бокс
        shutil.copy(os.path.join(src_images_dir, img_file), os.path.join(dst_img_dir, img_file))
        shutil.copy(os.path.join(src_labels_dir, label_file), os.path.join(dst_box_dir, label_file))

    print("Данные успешно разделены на train, val и test.")
    return True


def generate_masks_with_sam(split_name):
    images_dir = os.path.join(TARGET_BASE_DIR, split_name, "images")
    labels_box_dir = os.path.join(TARGET_BASE_DIR, split_name, "labels_box")
    labels_seg_dir = os.path.join(TARGET_BASE_DIR, split_name, "labels")

    os.makedirs(labels_seg_dir, exist_ok=True)

    print(f"\n[{split_name.upper()}] Инициализация SAM")
    sam_model = SAM("sam_b.pt").to("cuda")

    txt_files = [f for f in os.listdir(labels_box_dir) if f.endswith(".txt")]
    print(f"[{split_name.upper()}] Начинаем пакетную генерацию для {len(txt_files)} файлов.")

    for count, filename in enumerate(txt_files, 1):
        box_file_path = os.path.join(labels_box_dir, filename)
        img_file_path = os.path.join(images_dir, filename.replace(".txt", ".jpg"))
        if not os.path.exists(img_file_path):
            img_file_path = os.path.join(images_dir, filename.replace(".txt", ".png"))

        image = cv2.imread(img_file_path)
        if image is None:
            continue
        h, w, _ = image.shape

        bboxes_list = []
        class_ids = []

        with open(box_file_path, "r") as f:
            lines = f.readlines()

        # Шаг 1: Сначала просто собираем ВСЕ рамки для текущей картинки в один список
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_ids.append(parts[0])
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bboxes_list.append([x1, y1, x2, y2])

        # Если на листе вообще нет боксов, создаем пустой файл
        if not bboxes_list:
            with open(os.path.join(labels_seg_dir, filename), "w") as f:
                f.write("")
            continue

        new_segment_lines = []

        # Шаг 2: Передаем в SAM картинку и список всех боксов
        sam_results = sam_model(img_file_path, bboxes=bboxes_list, verbose=False, device="cuda")

        # Шаг 3: Разбираем результаты пакетного инференса
        if sam_results and len(sam_results) > 0:
            result_entry = sam_results[0]  # Достаем объект результатов для этой картинки

            if result_entry.masks is not None:
                # В пакетном режиме .xyn содержит маски для каждого переданного бокса по очереди
                all_polygons = result_entry.masks.xyn

                for idx, class_id in enumerate(class_ids):
                    if idx < len(all_polygons):
                        single_object_polygon = all_polygons[idx]
                        if len(single_object_polygon) > 0:
                            coords_str = " ".join(
                                [f"{coord:.6f}" for point in single_object_polygon for coord in point])
                            new_segment_lines.append(f"{class_id} {coords_str}")

        # Шаг 4: Записываем готовые полигоны сегментации
        with open(os.path.join(labels_seg_dir, filename), "w") as f:
            f.write("\n".join(new_segment_lines))

        if count % 50 == 0 or count == len(txt_files):
            print(f"[{split_name.upper()}] Картинок обработано: {count}/{len(txt_files)}")


if __name__ == "__main__":
    # 1. Сначала разделяем данные на train, val и test
    success = split_and_copy_data(val_ratio=0.15, test_ratio=0.15)

    # 2. Если разделение прошло успешно, запускаем SAM маскирование для ВСЕХ трех папок
    if success:
        generate_masks_with_sam("train")
        generate_masks_with_sam("val")
        generate_masks_with_sam("test")
        print("Датасет готов")
