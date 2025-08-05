import customtkinter as ctk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

def pca_manual(X, n_components):
    # Tính trung bình và độ lệch chuẩn theo từng cột
    mean = np.mean(X, axis=0)
    
    std = np.std(X, axis=0)
    
    X_standardized = (X - mean) / std

    # Tính ma trận hiệp phương sai
    cov_matrix = np.cov(X_standardized, rowvar=False)

    # Tính trị riêng và vector riêng
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

    # Sắp xếp trị riêng giảm dần và chọn vector tương ứng
    sorted_idx = np.argsort(eigen_values)[::-1]
    
    top_vectors = eigen_vectors[:, sorted_idx[:n_components]]

    # Giảm chiều dữ liệu, chiếu dữ liệu lên vector cơ sở mới 
    X_reduced = np.dot(X_standardized, top_vectors)

    # Trả về thêm mean và std để chuẩn hóa dữ liệu mới
    return X_reduced, top_vectors, eigen_values[sorted_idx], mean, std

class PCAApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("✨ PCA GUI - CustomTkinter (Manual PCA + Suggestion + Variance Plot)")
        self.geometry("1250x1150")

        self.data = None
        self.pca_result = None
        self.labels = None
        self.file_path = None

        self.build_gui()

    def build_gui(self):
        self.frame_top = ctk.CTkFrame(self)
        self.frame_top.pack(pady=10, padx=10, fill="x")

        self.label_info = ctk.CTkLabel(self.frame_top, text="📂 Chọn file CSV dữ liệu:")
        self.label_info.grid(row=0, column=0, padx=10)

        self.btn_load = ctk.CTkButton(self.frame_top, text="Tải dữ liệu", command=self.load_data)
        self.btn_load.grid(row=0, column=1, padx=5)

        self.btn_run = ctk.CTkButton(self.frame_top, text="🚀 Giảm chiều (PCA chay)", command=self.run_pca, state="disabled")
        self.btn_run.grid(row=0, column=2, padx=5)

        self.label_dim = ctk.CTkLabel(self.frame_top, text="🔢 Số chiều mong muốn:")
        self.label_dim.grid(row=0, column=3, padx=10)

        self.entry_dim = ctk.CTkEntry(self.frame_top, width=60)
        self.entry_dim.insert(0, "0")
        self.entry_dim.grid(row=0, column=4)

        self.text_output = ctk.CTkTextbox(self, height=140, width=1200, font=("Consolas", 12))
        self.text_output.pack(pady=10, padx=10)

        self.frame_plot = ctk.CTkFrame(self)
        self.frame_plot.pack()

        self.fig2d, self.ax2d = plt.subplots(figsize=(5.5, 4))
        self.canvas2d = FigureCanvasTkAgg(self.fig2d, master=self.frame_plot)
        self.canvas2d.get_tk_widget().grid(row=0, column=0, padx=10)

        self.fig3d = plt.figure(figsize=(5.5, 4))
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=self.frame_plot)
        self.canvas3d.get_tk_widget().grid(row=0, column=1, padx=10)

        self.frame_bottom = ctk.CTkFrame(self)
        self.frame_bottom.pack(pady=10)

        self.fig_var, self.ax_var = plt.subplots(figsize=(6, 3))
        self.canvas_var = FigureCanvasTkAgg(self.fig_var, master=self.frame_bottom)
        self.canvas_var.get_tk_widget().grid(row=0, column=0, padx=10)

        self.fig_err, self.ax_err = plt.subplots(figsize=(6, 3))
        self.canvas_err = FigureCanvasTkAgg(self.fig_err, master=self.frame_bottom)
        self.canvas_err.get_tk_widget().grid(row=0, column=1, padx=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            self.data = pd.read_csv(file_path)
            self.file_path = file_path
            self.btn_run.configure(state="normal")
            self.text_output.delete("0.0", "end")
            self.text_output.insert("0.0", f"📁 Đã tải: {os.path.basename(file_path)}\n")

            numeric_data = self.data.select_dtypes(include=[np.number])
            _, _, eigenvals, _, _ = pca_manual(numeric_data.values, numeric_data.shape[1])
            cumulative = np.cumsum(eigenvals) / np.sum(eigenvals)
            suggested_dim = np.argmax(cumulative >= 0.95) + 1
            variance = cumulative[suggested_dim - 1]
            self.text_output.insert("end", f"🔍 Gợi ý: Nên chọn {suggested_dim} chiều để giữ lại {variance:.2%} phương sai\n")

            self.ax_var.clear()
            self.ax_var.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
            self.ax_var.axhline(y=0.95, color="r", linestyle="--", label="95%")
            self.ax_var.set_title("Biểu đồ phương sai tích lũy theo số chiều")
            self.ax_var.set_xlabel("Số chiều")
            self.ax_var.set_ylabel("Phương sai tích lũy")
            self.ax_var.grid(True)
            self.ax_var.legend()
            self.canvas_var.draw()

        except Exception as e:
            self.text_output.insert("0.0", f"❌ Lỗi khi đọc file: {e}\n")

    def run_pca(self):
        try:
            numeric_data = self.data.select_dtypes(include=[np.number])
            self.labels = self.data['label'] if 'label' in self.data.columns else None
            X_original = numeric_data.values

            n_dim = int(self.entry_dim.get())
            _, _, eigenvals, _, _ = pca_manual(X_original, numeric_data.shape[1])
            cumulative = np.cumsum(eigenvals) / np.sum(eigenvals)

            if n_dim <= 0:
                n_dim = np.argmax(cumulative >= 0.95) + 1
            variance = np.sum(eigenvals[:n_dim]) / np.sum(eigenvals)

            self.pca_result, eigvecs, _, mean_vec, std_vec = pca_manual(X_original, max(3, n_dim))
            X_standardized = (X_original - mean_vec) / std_vec
            X_reconstructed = np.dot(self.pca_result[:, :n_dim], eigvecs[:, :n_dim].T) * std_vec + mean_vec
            error = np.mean((X_original - X_reconstructed) ** 2)

            errors = []
            dims = range(1, min(30, numeric_data.shape[1]) + 1)
            for d in dims:
                X_red, eigv, _, mean, std = pca_manual(X_original, d)
                X_rec = np.dot(X_red, eigv.T) * std + mean
                mse = np.mean((X_original - X_rec) ** 2)
                errors.append(mse)

            output_path = os.path.join(os.path.dirname(self.file_path), "reduced_output.csv")
            pd.DataFrame(self.pca_result).to_csv(output_path, index=False)
            input_size = os.path.getsize(self.file_path) / 1024
            output_size = os.path.getsize(output_path) / 1024
            compression = (1 - output_size / input_size) * 100

            self.text_output.delete("0.0", "end")
            self.text_output.insert("0.0", f"Số chiều ban đầu: {numeric_data.shape[1]}\n")
            self.text_output.insert("end", f"Số chiều đã chọn: {n_dim} {'(Tự động)' if int(self.entry_dim.get()) <= 0 else '(Thủ công)'}\n")
            self.text_output.insert("end", f"Tỷ lệ phương sai giữ lại: {variance:.4f}\n")
            self.text_output.insert("end", f"🔁 Sai số tái tạo (MSE): {error:.6f}\n")
            self.text_output.insert("end", f"Kích thước file gốc: {input_size:.2f} KB\n")
            self.text_output.insert("end", f"Kích thước sau PCA: {output_size:.2f} KB\n")
            self.text_output.insert("end", f"📉 Giảm được: {compression:.2f}%\n")

            self.plot_result()
            self.plot_error_chart(dims, errors)

        except Exception as e:
            self.text_output.insert("0.0", f"❌ Lỗi PCA: {e}\n")

    def plot_result(self):
        if self.pca_result is None:
            return

        self.ax2d.clear()
        if self.labels is not None:
            for lbl in np.unique(self.labels):
                idx = self.labels == lbl
                self.ax2d.scatter(self.pca_result[idx, 0], self.pca_result[idx, 1], label=f"Label {lbl}", alpha=0.7)
            self.ax2d.legend()
        else:
            self.ax2d.scatter(self.pca_result[:, 0], self.pca_result[:, 1], alpha=0.7)
        self.ax2d.set_title("PCA 2D")
        self.ax2d.set_xlabel("PC1")
        self.ax2d.set_ylabel("PC2")
        self.canvas2d.draw()

        self.ax3d.clear()
        if self.pca_result.shape[1] >= 3:
            if self.labels is not None:
                for lbl in np.unique(self.labels):
                    idx = self.labels == lbl
                    self.ax3d.scatter(self.pca_result[idx, 0], self.pca_result[idx, 1], self.pca_result[idx, 2], label=f"Label {lbl}", alpha=0.7)
                self.ax3d.legend()
            else:
                self.ax3d.scatter(self.pca_result[:, 0], self.pca_result[:, 1], self.pca_result[:, 2], alpha=0.7)
            self.ax3d.set_title("PCA 3D")
            self.ax3d.set_xlabel("PC1")
            self.ax3d.set_ylabel("PC2")
            self.ax3d.set_zlabel("PC3")
        else:
            self.ax3d.text2D(0.3, 0.5, "Không đủ chiều để vẽ 3D", transform=self.ax3d.transAxes)
        self.canvas3d.draw()

    def plot_error_chart(self, dims, errors):
        self.ax_err.clear()
        self.ax_err.plot(dims, errors, marker="o")
        self.ax_err.set_title("Biểu đồ sai số tái tạo theo số chiều")
        self.ax_err.set_xlabel("Số chiều")
        self.ax_err.set_ylabel("Sai số (MSE)")
        self.ax_err.grid(True)
        self.canvas_err.draw()

if __name__ == "__main__":
    app = PCAApp()
    app.mainloop()
