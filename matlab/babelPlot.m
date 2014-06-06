figure('Color', [1 1 1]);

plot(x, y, '-ob', 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerEdgeColor', [.5, 0, .5], 'MarkerFaceColor', [.5, 0, .5])

%plot(x, y, '-ob', 'LineSmoothing', 'on', 'LineWidth', 1, 'MarkerSize', 7, 'MarkerEdgeColor', [.5, 0, .5], 'MarkerFaceColor', [.5, 0, .5])

xlabel('Dimension of random features', 'Fontname', 'Segoe UI', 'FontSize', 12)
ylabel('Percentage error', 'Fontname', 'Segoe UI', 'FontSize', 12)
title('Average percentage error', 'Fontsize', 13)
ylabel('Absolute error', 'Fontname', 'Segoe UI', 'FontSize', 12)
title('Average absolute error', 'Fontsize', 13)

set(gca, 'XTick', x(3:end));

