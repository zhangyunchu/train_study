# ui.py
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import torch
from model import (device, classes, train_loader, test_loader, load_model, train_model, test_model)

# ===============================
# Tkinter 手写识别界面
# ===============================
class EMNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写字母与数字识别系统")
        self.root.geometry("950x800")

        self.canvas_width = 280
        self.canvas_height = 280
        self.line_width = 30
        self.last_x, self.last_y = None, None

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)

        # 模型加载
        self.model_path = "./model/emnist_balanced_model.pkl"
        self.model = load_model(self.model_path)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.setup_ui()

    # ---------------------------
    # UI 布局
    # ---------------------------
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="手写字母与数字识别系统",
                  font=("Arial", 18, "bold")).pack(pady=10)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧画布
        left_frame = ttk.LabelFrame(content_frame, text="手写输入", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="清空画布", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="识别数字", command=lambda: self.recognize("digit")).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="识别字母", command=lambda: self.recognize("letter")).pack(side=tk.LEFT, padx=5)

        # 右侧控制区
        right_frame = ttk.LabelFrame(content_frame, text="控制面板", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Button(right_frame, text="训练模型", command=self.train_model_thread).pack(fill=tk.X, pady=5)
        ttk.Button(right_frame, text="测试模型", command=self.test_model_thread).pack(fill=tk.X, pady=5)
        ttk.Button(right_frame, text="保存图片", command=self.save_image).pack(fill=tk.X, pady=5)
        ttk.Button(right_frame, text="加载图片", command=self.load_image).pack(fill=tk.X, pady=5)

        ttk.Label(right_frame, text="进度:").pack(anchor="w", pady=(10, 0))
        self.progress = ttk.Progressbar(right_frame, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(pady=5)

        # 输出
        self.result_text = tk.Text(main_frame, height=10, width=80)
        self.result_text.pack(pady=10)

        self.status_var = tk.StringVar(value="系统就绪")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X)

    # ---------------------------
    # 绘图与清空
    # ---------------------------
    def paint(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill="black", width=self.line_width, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                           fill=0, width=self.line_width)
        self.last_x, self.last_y = event.x, event.y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_text.delete(1.0, tk.END)
        self.status_var.set("画布已清空")

    # ---------------------------
    # 图像预处理
    # ---------------------------
    def preprocess_image(self, image):
        image = image.convert("L")
        image = ImageOps.invert(image)
        image = ImageOps.autocontrast(image)
        image = image.point(lambda x: 0 if x < 128 else 255, '1').convert('L')

        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
        image.thumbnail((20, 20), Image.LANCZOS)

        new_img = Image.new("L", (28, 28), 0)
        new_img.paste(image, ((28 - image.width)//2, (28 - image.height)//2))
        new_img = new_img.rotate(-90, expand=False)
        new_img = ImageOps.mirror(new_img)

        img_array = np.array(new_img, dtype=np.float32) / 255.0
        img_array = (img_array - 0.1307) / 0.3081
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)
        return tensor

    # ---------------------------
    # 识别逻辑（只显示最高置信度）
    # ---------------------------
    def recognize(self, mode="digit"):
        try:
            self.status_var.set(f"正在识别 ({'数字' if mode=='digit' else '字母'})...")
            self.root.update()

            tensor = self.preprocess_image(self.image)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)

            filtered_indices = [
                i for i, label in enumerate(classes)
                if (mode == "digit" and label.isdigit()) or (mode == "letter" and label.isalpha())
            ]

            probs_filtered = probs[:, filtered_indices]
            top_prob, top_idx = torch.max(probs_filtered, 1)
            real_idx = filtered_indices[top_idx.item()]

            label = classes[real_idx]
            confidence = top_prob.item() * 100

            result = f"识别结果：{label}\n置信度：{confidence:.2f}%"
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result)
            self.status_var.set("识别完成")

        except Exception as e:
            messagebox.showerror("识别错误", str(e))
            self.status_var.set(" 识别失败")

    # ---------------------------
    # 图片操作
    # ---------------------------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.bmp")])
        if not path:
            return
        img = Image.open(path).convert("L").resize((self.canvas_width, self.canvas_height))
        self.image = img
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete("all")
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        self.status_var.set(f"已加载图片: {os.path.basename(path)}")

    def save_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if not path:
            return
        self.image.save(path)
        self.status_var.set(f"已保存图片: {path}")

    # ---------------------------
    # 模型训练与测试
    # ---------------------------
    def train_model_thread(self):
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        self.status_var.set("正在训练模型...")
        self.progress["value"] = 0
        def progress_cb(v): self.progress["value"] = v; self.root.update_idletasks()
        def epoch_cb(e, loss): self.result_text.insert(tk.END, f"Epoch {e} 完成，平均损失: {loss:.4f}\n")

        train_model(self.model, train_loader, self.criterion, self.optimizer,
                    self.model_path, progress_cb, epoch_cb)
        self.status_var.set("模型训练完成")
        self.progress["value"] = 100

    def test_model_thread(self):
        threading.Thread(target=self._test_thread, daemon=True).start()

    def _test_thread(self):
        self.status_var.set("正在测试模型...")
        self.progress["value"] = 0
        def progress_cb(v): self.progress["value"] = v; self.root.update_idletasks()

        acc = test_model(self.model, test_loader, progress_cb)
        self.result_text.insert(tk.END, f"测试集准确率: {acc*100:.2f}%\n")
        self.status_var.set(f"测试完成，准确率 {acc*100:.2f}%")
        self.progress["value"] = 100
