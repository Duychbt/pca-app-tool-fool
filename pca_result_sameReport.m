% --- PCA thủ công nhập dữ liệu từ bàn phím ---
clear; clc; close all;

%% --- Nhập kích thước ma trận ---
m = input('Nhập số hàng m: ');
n = input('Nhập số cột n: ');

%% --- Nhập dữ liệu ---
X = zeros(m, n);
disp('Nhập giá trị cho ma trận (theo từng phần tử):');
for i = 1:m
    for j = 1:n
        prompt = sprintf('Phần tử (%d,%d): ', i, j);
        X(i,j) = input(prompt);
    end
end

disp('--- Ma trận dữ liệu đã nhập ---');
disp(X);

%% --- Bước 1: Centering (trừ mean) ---
X_mean = mean(X);
X_centered = X - X_mean;

disp('--- Vector trung bình (mean) ---');
disp(X_mean);
disp('--- Ma trận đã trừ mean ---');
disp(X_centered);

%% --- Bước 2: Ma trận hiệp phương sai ---
S = (X_centered' * X_centered) / (m - 1);
disp('--- Ma trận hiệp phương sai ---');
disp(S);

%% --- Bước 3: Phân rã trị riêng ---
[eig_vecs, eig_vals] = eig(S);
eig_vals = diag(eig_vals);

%% --- Bước 4: Sắp xếp trị riêng giảm dần ---
[eig_vals_sorted, idx] = sort(eig_vals, 'descend');
eig_vecs_sorted = eig_vecs(:, idx);

disp('--- Trị riêng (Eigenvalues) ---');
disp(eig_vals_sorted);
disp('--- Vector riêng (Eigenvectors) ---');
disp(eig_vecs_sorted);

%% --- Bước 5: Tính phương sai giải thích ---
explained = eig_vals_sorted / sum(eig_vals_sorted) * 100;
cumulative = cumsum(explained);

disp('--- Phương sai giải thích (%) ---');
disp(explained');
disp('--- Phương sai tích lũy (%) ---');
disp(cumulative');

%% --- Bước 6: Chọn số chiều ---
disp('Chọn chế độ:');
disp('1. Thủ công');
disp('2. Tự động (>=95% phương sai)');
mode_choice = input('Lựa chọn (1 hoặc 2): ');

if mode_choice == 1
    k = input('Nhập số PC muốn lấy: ');
elseif mode_choice == 2
    k = find(cumulative >= 95, 1);
    fprintf('Chế độ tự động: chọn %d PC (>=95%% phương sai)\n', k);
else
    error('Lựa chọn không hợp lệ!');
end

%% --- Bước 7: Giảm chiều ---
Z = X_centered * eig_vecs_sorted(:, 1:k);

disp('--- Dữ liệu chiếu lên các PC đã chọn ---');
disp(Z);

%% --- Vẽ biểu đồ PCA 2D nếu k >= 2 ---
if k >= 2
    figure;
    scatter(Z(:,1), Z(:,2), 50, 'filled');
    xlabel('PC1'); ylabel('PC2');
    title('PCA - 2D Projection (Centering only)');
    grid on;
end

%% --- Vẽ biểu đồ phương sai tích lũy ---
figure;
plot(1:length(cumulative), cumulative, '-o', 'LineWidth', 2);
yline(95, '--r', '95% threshold');
xlabel('Số thành phần chính');
ylabel('Phương sai tích lũy (%)');
title('Cumulative Variance Explained');
grid on;
