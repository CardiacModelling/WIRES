close all
clear all

rng(10)

g=9.81;

x = [0:0.1:20]';
y = sqrt(2.0.*x./g);

figure
for i=1:4
subplot(2,2,i)
plot(x,y,'-','LineWidth',2,'Color',0.6*[1 1 1])
if (i==3 || i==4)
xlabel('Height (m)','FontSize',16)
end
if (i==1 || i==3)
ylabel('Time (s)','FontSize',16)
end
set(gca,'FontSize',14)
ylim([0,2])
hold on
end

x_sample = [8 9 10 11 12 13 14 15]';
y_sample = sqrt(2.0.*x_sample./g) + normrnd(0,0.1,length(x_sample),1);

% Now a straight line
subplot(2,2,2)
fitobject2 = fit(x_sample,y_sample,'poly1');
poly_predict2 = fitobject2(x);
plot(x, poly_predict2, '-','Color',[252 49 68]./256,'LineWidth',2)


% Overfit a spline
subplot(2,2,3)
fitobject = fit(x_sample,y_sample,'poly7')
poly_predict = fitobject(x);
plot(x, poly_predict, '-','Color',[82 216 105]./256,'LineWidth',2)

% Optimise g
subplot(2,2,4)
options = optimset('Display','Iter');
g_fit = fminsearch(@(x)objective_function(x_sample,y_sample,x),9,options);
y_fit = sqrt(2.0.*x./g_fit);
plot(x, y_fit,'--','LineWidth',2,'Color',[22 126 251]./256)
fprintf('We fitted gravitational acceleration g to be %g m/s^2\n',g_fit)

subfigs = {'A','B','C','D'};
for i=1:4
    subplot(2,2,i)
    text(1,1.75,subfigs{i},'FontSize',18)
    plot(x_sample, y_sample, 'o','MarkerSize',5,'LineWidth',3,'Color',[255 207 49]./256)
end

