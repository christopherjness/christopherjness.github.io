 clear; close all;
data = read_log(['run.log'],1)
 
 
 strain =  0.01*data.data(2:end,1);
 stress = -data.data(2:end,5)/(0.1*0.01);
 
 
 
figure
box on; hold on;
pbaspect([1 1 1])
%set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
set(gca,'Layer','top')
%xlim([0 1])
ylim([1 200]);
set(gca,'LineWidth',2,'FontSize',30);
%xlabel('Strain $\dot{\gamma}t$','FontSize',30,'interpreter','latex')
%ylabel('Viscosity $\sigma_{xy}/\eta_\mathrm{f}\dot{\gamma}$','FontSize',30,'interpreter','latex')


xxx = strain;
yyy = stress;
plot(xxx,yyy,'ok-','linewidth',2);
print('~/Desktop/Fig2','-depsc')


