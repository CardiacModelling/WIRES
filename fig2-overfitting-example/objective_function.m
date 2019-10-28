function err = objective_f(x_data,y_data,g)

y_fit = sqrt(2.0.*x_data./g);

err = sum((y_fit-y_data).^2);