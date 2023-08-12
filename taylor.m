close all;

%泰勒图简单的说就是一种可以表示标准差， 均方根误差和相关系数三个指标的图。比单一R?和RMSE等横纵坐标的图更加直观。
% 设置图框属性，包括图位置和尺寸
set(gcf,'units','inches','position',[0,8.0,12.0,8.0]);
set(gcf,'DefaultAxesFontSize',18); % 坐标轴字体大小
%读取数据，sd rmse 和 r方
data=xlsread('.\taylor.xls');%文件路径  

sdev = data(:,1);
crmsd = data(:,2);
ccoef = data(:,3);
%mmodel ID，我这里手动输入是因为要每个单独设置标志
ID = {'Obs','Seasonal','Trend','Remainder','Rainfall'};
label = ID;
%>>绘制 taylor_diagram
[hp, ht, axl] = taylor_diagram(sdev,crmsd,ccoef, ...
    'markerLabel',label, 'markerLegend', 'on', ...
    'markerDisplayed', 'colorBar', 'titleColorbar', 'RMSD', 'locationColorBar','EastOutside',  ...
    'styleSTD', '-','styleCOR', '--','colSTD','k', 'colCOR','k','colOBS','r', 'markerObs','o', ...
    'markerSize',15, 'tickRMS',[0:3:15],'limSTD',15, ...
    'tickRMSangle', 115, 'showlabelsRMS', 'on', ...
    'titleRMS','on', 'titleOBS','Observation');
% 保存文件
%writepng(gcf,'taylor fig.png');
saveas(gcf, 'taylor', 'png');