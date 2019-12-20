plot_init;
regression_methods = cell(2, 2);
regression_methods(1, :) = {@OLSwrapper, 'OLS'};
regression_methods(2, :) = {@MCWOLS, 'IWLS'};
f2 = @(x) 0.1*x.^4 - 0.5 * x + 10;
f2_int = @(x) 0.1/5 * x.^5 - 1/4 * x.^2 + 10*x;
noise_model = @(N) normrnd(0, 40, N, 1);
data_dist = @(N) gamrnd(2, 1, N, 1);
N = 500;
test_imbalanced_regression(f2, f2_int, noise_model, data_dist, ...
    regression_methods, N, 'results-static.csv');

fig = gcf;
xlabel('x');
ylabel('y');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 9);
annotation(gcf, 'textbox', [0 .95 .01 .01], 'String', 'A', 'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold')
print_fig_to_png(fig, '../static-results', 3, 2);

%% Function definitions

function [fun, theta, Sigma] = OLS(xs, ys, weights)
    if nargin < 3
        weights = ones(length(ys), 1);
    end
    sqrt_weights = sqrt(weights);
    theta = (sqrt_weights.*xs) \ (sqrt_weights.*ys);
    % See https://en.wikipedia.org/wiki/Ordinary_least_squares and
    % Faraway (2002): Practical Regression and Anova using R
    % for the calculation of the parameter estimate covariance Sigma.
    res = ys - xs * theta;
    err_var = res' * res / (length(ys) - length(theta));
    Sigma = err_var * inv(xs' * xs);    
    fun = @(x) theta(1) + x*theta(2:end);
end

function fun = OLSwrapper(xs, ys)
    fun = OLS([ones(length(ys), 1), xs], ys);
end

function fun = MCWOLS(xs, ys)
    % Perform monte carlo estimate of weights
    [N, p] = size(xs);
    xM = max(xs);
    xm = min(xs);
    Nsamples = N * 10;
    unif_samples = xm + (xM-xm).*rand(Nsamples, p);
    weights = zeros(length(ys), 1);
    for ii=1:Nsamples
        dist = sum(abs(unif_samples(ii, :) - xs), 2);
        [~, idx] = min(dist);
        weights(idx) = weights(idx) + 1;
    end
    weights = cap_weight_range(weights, inf);
    figure;
    plot(xs, weights, 'o');
    xlabel('x');
    ylabel('MCWOLS Weight');
    fun = OLS([ones(length(ys), 1), xs], ys, weights);
end

function capped_weights = cap_weight_range(weights, range)
    capped_weights = weights;
    wm = min(weights(weights~=0));
    max_weight = wm*range;
    capped_weights(weights > max_weight) = max_weight;
end