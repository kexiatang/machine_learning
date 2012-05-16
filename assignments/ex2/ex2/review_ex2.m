clear;close all; clc;

data = load('ex2data1.txt');
X = data(:,1:end-1);
y = data(:,end);


function plot_data(X,y)
pos = find(y==1);
neg = find(y==0);
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7)
end

figure; hold on;
plot_data(X,y);
hold off;

xlabel("value for x1");
ylabel("value for x2");

legend("x1","x2");

function g = sigmoid(z)
g = 1./(1+power(e,-z));
end

function [J, grad] = cost_function(theta, X, y)
m = length(y);
h = sigmoid(X * theta);
J = 1/m * sum(-y .* log(h)-(1-y).*log(1-h));
grad = 1/m * (X'*(h-y));
end

m = length(y);
X = [ones(m,1),X];
n = size(X,2);
init_theta = zeros(n,1);

[cost,grad] = cost_function(init_theta, X, y);

options = optimset('GradObj', 'on', 'MaxIter', 400)
[theta,cost] = fminunc(@(t)cost_function(t,X,y), init_theta, options);

function plot_decision_boundary(theta,X,y)
plot_x = [min(X(:,2)), max(X(:,2))];
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
plot(plot_x,plot_y, '-');
legend("y=1","y=0","decision boundary");
end

hold on;
plot_decision_boundary(theta,X,y);

function p = predict(theta, X)
p = floor(sigmoid(X*theta)+0.5);
end

p = predict(theta, X);
mean(double(p==y))*100;





