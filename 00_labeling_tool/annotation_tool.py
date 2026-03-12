from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Instala Pillow con: python -m pip install pillow")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [
    "red", "lime", "cyan", "yellow", "magenta", "orange",
    "blue", "green", "purple", "gold", "deep pink", "turquoise"
]


@dataclass
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_pixel(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        x1 = (self.x_center - self.width / 2) * img_w
        y1 = (self.y_center - self.height / 2) * img_h
        x2 = (self.x_center + self.width / 2) * img_w
        y2 = (self.y_center + self.height / 2) * img_h
        return x1, y1, x2, y2

    @staticmethod
    def from_pixel(
        class_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_w: int,
        img_h: int,
    ) -> "YoloBox":
        x1, x2 = sorted([max(0, x1), min(img_w, x2)])
        y1, y2 = sorted([max(0, y1), min(img_h, y2)])

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        nw = w / img_w
        nh = h / img_h

        return YoloBox(class_id, xc, yc, nw, nh)


class YoloEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Label Editor")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 700)

        self.images_dir: Optional[Path] = None
        self.labels_dir: Optional[Path] = None
        self.image_files: List[Path] = []
        self.index = 0

        self.class_names: List[str] = ["class_0"]

        self.current_image: Optional[Image.Image] = None
        self.current_photo = None
        self.current_boxes: List[YoloBox] = []

        self.selected_box_index: Optional[int] = None
        self.hover_box_index: Optional[int] = None

        self.show_boxes = tk.BooleanVar(value=True)
        self.selected_class = tk.StringVar(value="class_0")
        self.status_var = tk.StringVar(value="Abrí una carpeta de imágenes")

        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_w = 0
        self.display_h = 0

        self.drag_start: Optional[Tuple[int, int]] = None
        self.temp_rect = None

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Abrir imágenes", command=self.open_images_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Abrir labels", command=self.open_labels_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Cargar clases", command=self.load_classes_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Guardar", command=self.save_current_labels).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Anterior", command=self.prev_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Siguiente", command=self.next_image).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(top, text="Mostrar BB", variable=self.show_boxes, command=self.redraw).pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text="Clase activa:").pack(side=tk.LEFT, padx=(20, 4))
        self.class_combo = ttk.Combobox(
            top,
            textvariable=self.selected_class,
            state="readonly",
            width=25,
            values=self.class_names
        )
        self.class_combo.pack(side=tk.LEFT)

        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT, padx=8)

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Etiquetas", font=("Arial", 12, "bold")).pack(anchor="w")

        self.box_list = tk.Listbox(left, width=48, height=32)
        self.box_list.pack(fill=tk.Y, expand=True, pady=8)

        ttk.Button(left, text="Eliminar seleccionada", command=self.delete_selected_box).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Cambiar clase seleccionada", command=self.change_class).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Refrescar dibujo", command=self.redraw).pack(fill=tk.X, pady=4)

        help_text = (
            "Controles:\n"
            "- Arrastrá clic izquierdo para agregar BB\n"
            "- Click sobre una BB para seleccionarla\n"
            "- Pasá el mouse sobre una BB para resaltarla\n"
            "- Click derecho sobre una BB para eliminarla\n"
            "- Delete / Supr: eliminar BB seleccionada\n"
            "- C: cambiar clase de la BB seleccionada\n"
            "- H: mostrar / ocultar BB\n"
            "- Flechas izquierda/derecha: navegar\n"
            "- Ctrl+S: guardar"
        )
        ttk.Label(left, text=help_text, justify=tk.LEFT).pack(anchor="w", pady=10)

        self.canvas = tk.Canvas(main, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _bind_events(self):
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.delete_selected_box())
        self.root.bind("<Control-s>", lambda e: self.save_current_labels())
        self.root.bind("h", lambda e: self.toggle_boxes())
        self.root.bind("c", lambda e: self.change_class())

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)

        self.box_list.bind("<<ListboxSelect>>", self.on_list_select)

    def toggle_boxes(self):
        self.show_boxes.set(not self.show_boxes.get())
        self.redraw()

    def open_images_dir(self):
        path = filedialog.askdirectory(title="Seleccioná la carpeta de imágenes")
        if not path:
            return

        self.images_dir = Path(path)
        self.image_files = sorted(
            [p for p in self.images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        )

        if not self.image_files:
            messagebox.showerror("Error", "No se encontraron imágenes en la carpeta seleccionada.")
            return

        if self.labels_dir is None:
            self.labels_dir = self.images_dir

        self.index = 0
        self.load_image()

    def open_labels_dir(self):
        path = filedialog.askdirectory(title="Seleccioná la carpeta de labels YOLO")
        if not path:
            return

        self.labels_dir = Path(path)

        if self.image_files:
            self.load_image()

    def load_classes_file(self):
        path = filedialog.askopenfilename(
            title="Seleccioná classes.txt",
            filetypes=[("TXT", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]

        if not names:
            messagebox.showerror("Error", "El archivo de clases está vacío.")
            return

        self.class_names = names
        self.class_combo["values"] = self.class_names
        self.selected_class.set(self.class_names[0])
        self.refresh_box_list()
        self.redraw()

    def get_label_path(self, image_path: Path) -> Path:
        assert self.labels_dir is not None

        if self.images_dir is not None:
            rel = image_path.relative_to(self.images_dir)
            return (self.labels_dir / rel).with_suffix(".txt")

        return (self.labels_dir / image_path.name).with_suffix(".txt")

    def load_image(self):
        if not self.image_files:
            return

        image_path = self.image_files[self.index]

        try:
            self.current_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{image_path}\n\n{e}")
            return

        self.current_boxes = self.read_yolo_file(self.get_label_path(image_path))
        self.selected_box_index = None
        self.hover_box_index = None

        self.status_var.set(f"{self.index + 1}/{len(self.image_files)} - {image_path.name}")
        self.refresh_box_list()
        self.redraw()

    def read_yolo_file(self, path: Path) -> List[YoloBox]:
        boxes = []

        if not path.exists():
            return boxes

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    boxes.append(YoloBox(class_id, x, y, w, h))
                except ValueError:
                    continue

        return boxes

    def save_current_labels(self):
        if self.current_image is None or self.labels_dir is None or not self.image_files:
            return

        label_path = self.get_label_path(self.image_files[self.index])
        label_path.parent.mkdir(parents=True, exist_ok=True)

        with open(label_path, "w", encoding="utf-8") as f:
            for box in self.current_boxes:
                f.write(
                    f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} "
                    f"{box.width:.6f} {box.height:.6f}\n"
                )

        self.status_var.set(f"Guardado: {label_path.name}")

    def prev_image(self):
        if not self.image_files:
            return

        self.save_current_labels()
        self.index = max(0, self.index - 1)
        self.load_image()

    def next_image(self):
        if not self.image_files:
            return

        self.save_current_labels()
        self.index = min(len(self.image_files) - 1, self.index + 1)
        self.load_image()

    def redraw(self):
        self.canvas.delete("all")

        if self.current_image is None:
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        img_w, img_h = self.current_image.size

        self.scale = min(canvas_w / img_w, canvas_h / img_h)
        self.display_w = max(1, int(img_w * self.scale))
        self.display_h = max(1, int(img_h * self.scale))
        self.offset_x = (canvas_w - self.display_w) // 2
        self.offset_y = (canvas_h - self.display_h) // 2

        resized = self.current_image.resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.current_photo)

        if self.show_boxes.get():
            self.draw_boxes()

    def draw_boxes(self):
        if self.current_image is None:
            return

        img_w, img_h = self.current_image.size

        for i, box in enumerate(self.current_boxes):
            x1, y1, x2, y2 = box.to_pixel(img_w, img_h)
            sx1, sy1 = self.image_to_screen(x1, y1)
            sx2, sy2 = self.image_to_screen(x2, y2)

            color = COLORS[box.class_id % len(COLORS)]

            if i == self.selected_box_index:
                line_width = 4
            elif i == self.hover_box_index:
                line_width = 3
            else:
                line_width = 2

            self.canvas.create_rectangle(
                sx1, sy1, sx2, sy2,
                outline=color,
                width=line_width
            )

            if 0 <= box.class_id < len(self.class_names):
                class_name = self.class_names[box.class_id]
            else:
                class_name = f"class_{box.class_id}"

            label = f"{i}: {class_name}"
            if i == self.selected_box_index:
                label += " [SELECTED]"
            elif i == self.hover_box_index:
                label += " [HOVER]"

            text_w = max(120, 8 * len(label))
            y_top = max(0, sy1 - 20)

            self.canvas.create_rectangle(
                sx1, y_top, sx1 + text_w, sy1,
                fill=color,
                outline=color
            )
            self.canvas.create_text(
                sx1 + 4,
                y_top + 10,
                text=label,
                fill="black",
                anchor="w",
                font=("Arial", 10, "bold")
            )

    def refresh_box_list(self):
        self.box_list.delete(0, tk.END)

        for i, box in enumerate(self.current_boxes):
            if 0 <= box.class_id < len(self.class_names):
                class_name = self.class_names[box.class_id]
            else:
                class_name = f"class_{box.class_id}"

            text = (
                f"[{i}] {class_name} | xc={box.x_center:.3f}, yc={box.y_center:.3f}, "
                f"w={box.width:.3f}, h={box.height:.3f}"
            )
            self.box_list.insert(tk.END, text)

    def on_list_select(self, _event=None):
        sel = self.box_list.curselection()
        if not sel:
            return

        self.selected_box_index = sel[0]
        self.redraw()

    def on_mouse_move(self, event):
        if not self.show_boxes.get():
            return

        hit_idx = self.find_box_at_screen(event.x, event.y)
        if hit_idx != self.hover_box_index:
            self.hover_box_index = hit_idx
            self.redraw()

    def on_mouse_leave(self, _event):
        if self.hover_box_index is not None:
            self.hover_box_index = None
            self.redraw()

    def on_left_down(self, event):
        if self.current_image is None:
            return

        hit_idx = self.find_box_at_screen(event.x, event.y)
        if hit_idx is not None:
            self.selected_box_index = hit_idx
            self.refresh_selection_in_list()
            self.redraw()
            return

        img_pt = self.screen_to_image(event.x, event.y)
        if img_pt is None:
            return

        self.drag_start = (event.x, event.y)
        self.temp_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="white",
            dash=(4, 4),
            width=2
        )

    def on_left_drag(self, event):
        if self.drag_start is None or self.temp_rect is None:
            return

        self.canvas.coords(
            self.temp_rect,
            self.drag_start[0], self.drag_start[1],
            event.x, event.y
        )

    def on_left_up(self, event):
        if self.drag_start is None or self.current_image is None:
            return

        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None

        if self.temp_rect is not None:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

        p0 = self.screen_to_image(x0, y0)
        p1 = self.screen_to_image(x1, y1)
        if p0 is None or p1 is None:
            return

        ix0, iy0 = p0
        ix1, iy1 = p1

        if abs(ix1 - ix0) < 4 or abs(iy1 - iy0) < 4:
            return

        class_name = self.selected_class.get()
        class_id = self.class_names.index(class_name) if class_name in self.class_names else 0

        img_w, img_h = self.current_image.size
        new_box = YoloBox.from_pixel(class_id, ix0, iy0, ix1, iy1, img_w, img_h)

        self.current_boxes.append(new_box)
        self.selected_box_index = len(self.current_boxes) - 1
        self.hover_box_index = None

        self.refresh_box_list()
        self.refresh_selection_in_list()
        self.redraw()

    def on_right_click(self, event):
        hit_idx = self.find_box_at_screen(event.x, event.y)
        if hit_idx is not None:
            self.selected_box_index = hit_idx
            self.refresh_selection_in_list()
            self.delete_selected_box()

    def delete_selected_box(self):
        if self.selected_box_index is None:
            return

        if not (0 <= self.selected_box_index < len(self.current_boxes)):
            self.selected_box_index = None
            return

        del self.current_boxes[self.selected_box_index]
        self.selected_box_index = None
        self.hover_box_index = None

        self.refresh_box_list()
        self.redraw()

    def change_class(self):
        if self.selected_box_index is None:
            return

        if not (0 <= self.selected_box_index < len(self.current_boxes)):
            return

        class_name = self.selected_class.get()
        if class_name not in self.class_names:
            return

        new_class_id = self.class_names.index(class_name)
        self.current_boxes[self.selected_box_index].class_id = new_class_id

        self.refresh_box_list()
        self.refresh_selection_in_list()
        self.redraw()

    def refresh_selection_in_list(self):
        self.box_list.selection_clear(0, tk.END)

        if self.selected_box_index is not None and 0 <= self.selected_box_index < self.box_list.size():
            self.box_list.selection_set(self.selected_box_index)
            self.box_list.see(self.selected_box_index)

    def image_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

    def screen_to_image(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        if self.current_image is None:
            return None

        ix = (x - self.offset_x) / self.scale
        iy = (y - self.offset_y) / self.scale

        img_w, img_h = self.current_image.size
        if ix < 0 or iy < 0 or ix > img_w or iy > img_h:
            return None

        return ix, iy

    def find_box_at_screen(self, sx: int, sy: int) -> Optional[int]:
        if self.current_image is None:
            return None

        img_pt = self.screen_to_image(sx, sy)
        if img_pt is None:
            return None

        x, y = img_pt
        img_w, img_h = self.current_image.size

        for i in reversed(range(len(self.current_boxes))):
            x1, y1, x2, y2 = self.current_boxes[i].to_pixel(img_w, img_h)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i

        return None


if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    app = YoloEditorApp(root)
    root.mainloop()