 clear; close all;
data = read_log(['log.lammps'],1)
 
 
 strain =  0.01*data.data(2:end,1);
 stress = -data.data(2:end,5)/(0.1*0.01);
 
 
 
figure
box on; hold on;
%set(gca, 'XScale', 'log')
%set(gca, 'YScale', 'log')
set(gca,'Layer','top')
%xlim([0 1])
%ylim([0.1 4]);
set(gca,'LineWidth',2,'FontSize',30);
xlabel('Strain $\dot{\gamma}t$','FontSize',30,'interpreter','latex')
ylabel('Viscosity $\sigma/\eta\dot{\gamma}$','FontSize',30,'interpreter','latex')


xxx = strain;
yyy = stress;
plot(xxx,yyy,'-o','linewidth',2);

