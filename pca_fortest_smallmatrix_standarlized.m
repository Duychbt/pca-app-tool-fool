
clear; clc; close all;

%% --- Bước 1: Nhập dữ liệu ---
m = input('Nhập số hàng m: ');
n = input('Nhập số cột n: ');

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

%% --- Bước 2: Chuẩn hóa dữ liệu (Standardization) ---
X_mean = mean(X);
X_stddev = std(X);
X_std = (X - X_mean) ./ X_stddev;  % chuẩn hóa: mean=0, std=1

%% --- Bước 3: Ma trận hiệp phương sai ---
cov_matrix = cov(X_std);

%% --- Bước 4: Phân rã trị riêng ---
[eig_vecs, eig_vals] = eig(cov_matrix);
eig_vals = diag(eig_vals);

%% --- Bước 5: Sắp xếp giảm dần ---
[eig_vals_sorted, idx] = sort(eig_vals, 'descend');
eig_vecs_sorted = eig_vecs(:, idx);

%% --- Bước 6: Variance explained ---
explained = eig_vals_sorted / sum(eig_vals_sorted) * 100;
cumulative = cumsum(explained);

disp('--- Phương sai giải thích (%) của từng PC ---');
disp(explained');
disp('--- Phương sai tích lũy (%) ---');
disp(cumulative');

%% --- Bước 7: Chọn số chiều ---
disp('Chọn số chiều:');
disp('1. Tự động (>=95% variance)');
disp('2. Nhập thủ công');
choice = input('Lựa chọn (1 hoặc 2): ');

if choice == 1
    k = find(cumulative >= 95, 1); % tự động
else
    k = input('Nhập số chiều muốn giảm còn: ');
end

disp(['--- Số chiều được chọn: ', num2str(k)]);

%% --- Bước 8: Giảm chiều ---
X_reduced = X_std * eig_vecs_sorted(:, 1:k);
disp('--- Dữ liệu sau khi giảm chiều ---');
disp(X_reduced);

%% --- Bước 9: Vẽ biểu đồ ---
figure;
plot(1:length(cumulative), cumulative, '-o', 'LineWidth', 2);
yline(95, '--r', '95% threshold');
xlabel('Số thành phần chính');
ylabel('Phương sai tích lũy (%)');
title('PCA - Cumulative Variance Explained');
grid on;

if k >= 2
    figure;
    scatter(X_reduced(:,1), X_reduced(:,2), 50, 'filled');
    xlabel('PC1');
    ylabel('PC2');
    title('PCA - 2D Projection');
    grid on;
end
