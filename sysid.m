% Add online KDE estimation code
% This requires the maggot toolbox to be installed, which can be obtained
% here: http://www.vicos.si/Research/Multivariate_Online_Kernel_Density_Estimation
addpath(genpath('maggot/'));

rng default;
dT = 0.01;
t = 0:dT:60;
f = 1 + 0.1 * sin(2*pi*t/60);
u = sin(pi*t.*f);
u(t >= 38.5 & t < 40.5) = -4;
u(t >= 40.5 & t < 42.5) = 4;
u_nom = movmean(u, 200);
u = u_nom + 0.05 * randn(size(u_nom));

a2 = 2;
a1 = 4*a2; % enforce minimum of quadratic relation at u=-2
a3 = -a2; % enforce f(-1)=-f(1)

N = length(t);
xs = zeros(2, N);
TANH_SCALE = 0.2;
% ufun = @(u) (a1*u + a2*u.^2 + a3) .* dT;
ufun = @(u) u * dT;
x1_nom = @(x, u) 0.98*x(1) + 0.05*x(2) + ufun(u);
x2_nom = @(x, u) -tanh(x(1)/TANH_SCALE) * dT + 0.98 * x(2);
proc_noise = [0.001; 0.001] .* randn(2, N-1);

for kk = 2:N
    xs(1, kk) = x1_nom(xs(:, kk-1), u(kk-1)) + proc_noise(1, kk-1);
    xs(2, kk) = x2_nom(xs(:, kk-1), u(kk-1)) + proc_noise(2, kk-1);
end

%% Calculate weights
% First define the cost function: MSE wrt the best estimate
x1min = min(xs(1, :));
x1max = max(xs(1, :));
x1test = x1min:.01:x1max;
cross_true = -tanh(x1test/TANH_SCALE) * dT;
pows = @(x) [x x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 x.^8];
reg_mat = [pows(x1test') ones(length(x1test), 1)];
best_params = reg_mat \ cross_true';
crossfun_best = best_params' * reg_mat';
x1_delayed = xs(1, 1:N-1)';
x2_delayed = xs(2, 1:N-1)';
x1_delayed_pow = pows(x1_delayed);
x2_delayed_pow = pows(x2_delayed);
target = xs(:, 2:N)';
regression_matrix_x1 = [x1_delayed x2_delayed_pow u(1:N-1)' ones(N-1, 1)];
regression_matrix_x2 = [x1_delayed_pow x2_delayed u(1:N-1)' ones(N-1, 1)];

% --- Random Search for ideal oKDE parameter Dth ---
% function loss = mse(weights)
%     params_x2_weighted = lasso(regression_matrix_x2, target(:, 2), 'Lambda', 0, 'Weights', weights);
%     crossfun_est = params_x2_weighted([1:8,11])' * reg_mat';
%     assert(all(size(crossfun_est) == size(crossfun_best)));
%     loss = mean((crossfun_best - crossfun_est).^2);
% end
% rng default % For reproducibility
% lb = 0.01;
% ub = 0.1;
% costfun = @(param) mse(calc_weights([xs(:, 1:N-1)' u(1:N-1)'], param));
% tic;
% min_cost = inf;
% opt_params = nan;
% ii = 1;
% elapsed = 0;
% while elapsed < 7000
%     param = lb + (ub-lb) .* rand(1);
%     cost = costfun(param);
%     if cost < min_cost
%         opt_param = param;
%         min_cost = cost;
%     end
%     elapsed = toc;
%     fprintf('Iteration %d (%f s elapsed): Dth=%f\n', ii, elapsed, opt_params);
%     ii = ii + 1;
% end

% Calculate weights
opt_param = 0.071086;
weights = calc_weights([xs(:, 1:N-1)' u(1:N-1)'], opt_param);

%% Perform regression with and without weights
disp('Params without weighting:');
% L2 regression for x1
params_x1_unweighted = lasso(regression_matrix_x1, target(:, 1), 'Lambda', 0);
params_x2_unweighted = lasso(regression_matrix_x2, target(:, 2), 'Lambda', 0);
disp([params_x1_unweighted params_x2_unweighted]);

disp('Params with oKDE weighting:');
% weighted L2 regression for x1
params_x1_weighted = lasso(regression_matrix_x1, target(:, 1), 'Lambda', 0, 'Weights', weights);
% weighted L2 regression for x2
params_x2_weighted = lasso(regression_matrix_x2, target(:, 2), 'Lambda', 0, 'Weights', weights);
disp([params_x1_weighted params_x2_weighted]);

% Use identified parameters to predict states
nsteps = 100;
xs_pred_unweighted = zeros(size(xs));
x1_pred_unweighted = @(x, u) params_x1_unweighted' * [x(1)'; pows(x(2))'; u; 1];
x2_pred_unweighted = @(x, u) params_x2_unweighted' * [pows(x(1))'; x(2)'; u; 1];
xs_pred_weighted = zeros(size(xs));
x1_pred_weighted = @(x, u) params_x1_weighted' * [x(1)'; pows(x(2))'; u; 1];
x2_pred_weighted = @(x, u) params_x2_weighted' * [pows(x(1))'; x(2)'; u; 1];
for kk = 1+nsteps:N
    x_pred_unweighted = xs(:, kk-nsteps);
    x_pred_weighted = xs(:, kk-nsteps);
    for jj = 1:nsteps
        ucurr = u_nom(kk-nsteps+jj-1);
        x_pred_unweighted = [x1_pred_unweighted(x_pred_unweighted, ucurr); x2_pred_unweighted(x_pred_unweighted, ucurr)];
        x_pred_weighted = [x1_pred_weighted(x_pred_weighted, ucurr); x2_pred_weighted(x_pred_weighted, ucurr)];
    end
    xs_pred_unweighted(:, kk) = x_pred_unweighted;
    xs_pred_weighted(:, kk) = x_pred_weighted;
end

%% Plot everything
plot_init;
fig1 = figure;
ax1 = subplot(3, 1, 1);
plot(t, u);
ylabel('u')
ax2 = subplot(3, 1, 2);
plot(t, xs(1, :));
hold on;
plot(t, xs_pred_unweighted(1, :));
plot(t, xs_pred_weighted(1, :));
lgd = legend('Truth', 'OLS', 'IWLS', 'Location', 'northwest');
ylabel('x_1');
ax3 = subplot(3, 1, 3);
plot(t, xs(2, :));
hold on;
plot(t, xs_pred_unweighted(2, :));
plot(t, xs_pred_weighted(2, :));
ylabel('x_2');
xlabel('Time (s)');
set(findall(gcf,'-property', 'FontSize'), 'FontSize', 9);
annotation(gcf, 'textbox', [.05 .9 .1 .1], 'String', 'C', 'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold')
print_fig_to_png(fig1, '../sysid-signals', 6, 2);

fig2 = figure;
lwidth = 2;
scatter(xs(1, 1:N-1), -tanh(xs(1, 1:N-1)/TANH_SCALE) * dT + proc_noise(2, :), 25, 'p', 'filled');
hold on;
x1test = min(xs(1, :)):.01:max(xs(1, :));
cross_true = -tanh(x1test/TANH_SCALE) * dT;
reg_mat = [pows(x1test') ones(length(x1test), 1)];
crossfun_unweighted = @(x1) params_x2_unweighted(1:8)' * pows(x1')';
crossfun_weighted = @(x1) params_x2_weighted(1:8)' * pows(x1')';
plot(x1test, crossfun_unweighted(x1test), 'LineWidth', lwidth);
plot(x1test, crossfun_weighted(x1test), 'LineWidth', lwidth);
plot(x1test, crossfun_best, 'LineWidth', lwidth);
plot(x1test, cross_true, 'LineWidth', lwidth);
legend('Data', 'OLS', 'IWLS', 'BLE', 'f_{21}(x)');
xlabel('x_1');
ylabel('f_{21}(x_1)');
set(findall(gcf,'-property', 'FontSize'), 'FontSize', 9);
annotation(gcf, 'textbox', [0 .95 .01 .01], 'String', 'B', 'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold')
print_fig_to_png(fig2, '../sysid-results', 3, 2);

function weights = calc_weights(data, Dth)
% Calculate weights
probs = oKDE_probs(data, Dth);
weights = movmedian(1./probs, 100);
end