clear; close all; clc;
data = load("ex2data2.txt")
X = data(:, 1:end-1); y = data(:,end);

function plot_data(X,y)
pos = find(y==1);
neg = find(y==0);
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7)
end

hold on;
plot_data(X,y);

function out = map_features(X1,X2)
    degree = 6;
    out = ones(size(X1(:,1)));
    for i = 1:degree
        for j = 0:i
            out(:, end+1) = (X1.^(i-j)).*(X2.^j);
        end
    end
end
            
X = map_features(X(:,1),X(:,2));

m = length(y);
n = size(X,2)+1
init_theta = zeros(n,1);
lambda = 1;

X = [ones(m,1),X];

function g = sigmoid(z)
g = 1./(1+power(e,-z));
end

function [J, grad] = cost_function_reg(theta,X,y,lambda)
grad = zeros(size(theta));
m = length(y);
h = sigmoid(X * theta);
J = 1/m * sum(-y .* log(h) - (1-y) .* log(1-h)) + lambda/(2*m) * theta(2:end)' * theta(2:end);
grad(1) = 1/m * sum(h-y);
grad(2:end) = 1/m * (X(:, 2:end)' * (h-y)) + lambda/m * theta(2:end);
end

options = optimset("GradObj", "on", "MaxIter", 400)
[theta, cost] = fminunc(@(t)cost_function_reg(t,X,y,lambda), init_theta, options)

function plot_decision_boundary(theta, X, y)
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = [1, mapFeature(u(i), v(j))]*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0,0], 'LineWidth', 2)
end

plot_decision_boundary(theta, X, y);
