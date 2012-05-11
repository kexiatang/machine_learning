clear; clc; close;

% plot data

function plot_data(x,y)
    plot(x,y, 'rx', 'MarkerSize', 10);
    xlabel("feature");
    ylabel("target variable");
end
            
data = load("ex1data1.txt")
x = data(:,1)
y = data(:,2)
plot_data(x,y)


% compute cost
function H = hypothesis(theta, x)
    H = theta' * [1;x]
end

function J = compute_cost(X, y, theta)
    m = length(y);
    predict = X * theta;
    sqrErrors = (predict - y).^2;
    J = 1/(2*m) * sum(sqrErrors);
end

            
m = length(y);
n = 1;
X = [ones(m,1), x];
theta = zeros(n+1, 1)
J = compute_cost(X, y, theta)

function theta = gradiant_descent(X, y, theta, alpha)
    m = length(y);
    while true
        J1 = compute_cost(X, y, theta)
        predict = X * theta;
        diff = predict - y;
        delta = 1/m * X' * diff
        theta = theta - alpha * delta
        J2 = compute_cost(X, y, theta)
        if abs(J1-J2)<0.0001
                break;
        end;
    end
end

alpha = 0.01
theta = gradiant_descent(X, y, theta, alpha)

hold on;
plot(x, X*theta, '-');
legend('Training data', 'Linear regression')







