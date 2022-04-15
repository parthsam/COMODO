clc
clear all
close all


%%%%%%%%%%%DATA%%%%%%%%%

TFstv=importdata('STVTF.mat');
MASstv=importdata("STVMAS.mat");
EASstv=importdata('STVEAS.mat');

TFar=importdata('ARTF.mat');
MASar=importdata('ARMAS.mat');
EASar=importdata('AREAS.mat');


TFstvexpanded=importdata('STVTFexpanded.mat');
MASstvexpanded=importdata('STVMASexpanded.mat');
EASstvexpanded=importdata('STVEASexpanded.mat');

TFarexpanded=importdata('ARTFexpanded.mat');
MASarexpanded=importdata('ARMASexpanded.mat');
EASarexpanded=importdata('AREASexpanded.mat');

TFstv100replicas=importdata('STVTF100replicas.mat');
MASstv100replicas=importdata('STVMAS100replicas.mat');
EASstv100replicas=importdata('STVEAS100replicas.mat');

TFar100replicas=importdata('ARTF100replicas.mat');
MASar100replicas=importdata('ARMAS100replicas.mat');
EASar100replicas=importdata('AREAS100replicas.mat');


TFstv200replicas=importdata('STVTF200replicas.mat');
MASstv200replicas=importdata('STVMAS200replicas.mat');
EASstv200replicas=importdata('STVEAS200replicas.mat');

TFar200replicas=importdata('ARTF200replicas.mat');
MASar200replicas=importdata('ARMAS200replicas.mat');
EASar200replicas=importdata('AREAS200replicas.mat');

TPCTF=importdata('TPCTF.mat');
TPCMAS=importdata('TPCMAS.mat');
TPCEAS=importdata('TPCEAS.mat');


TPCTFexpanded=importdata('TPCTFexpanded.mat');
TPCMASexpanded=importdata('TPCMASexpanded.mat');
TPCEASexpanded=importdata('TPCEASexpanded.mat');


TPCTF100=importdata('TPCTF100replicas3D.mat');
TPCMAS100=importdata('TPCMAS100replicas3D.mat');
TPCEAS100=importdata('TPCEAS100replicas3D.mat');


TPCTF200=importdata('TPCTF200replicas3D.mat');
TPCMAS200=importdata('TPCMAS200replicas3D.mat');
TPCEAS200=importdata('TPCEAS200replicas3D.mat');

SD3D=importdata('3DProojection.txt');


Distance=SD3D;
figure()
hold on

% scatter3(Distance(1:4,1),Distance(1:4,2),Distance(1:4,3),'filled','b');
% scatter3(Distance(5:8,1),Distance(5:8,2),Distance(5:8,3),'filled','r');
% scatter3(Distance(9:12,1),Distance(9:12,2),Distance(9:12,3),'filled','g');
% scatter3(Distance(13:16,1),Distance(13:16,2),Distance(13:16,3),'filled','m');

% scatter3(Distance(1:20,1),Distance(1:20,2),Distance(1:20,3),'filled','r');
% scatter3(Distance(21:40,1),Distance(21:40,2),Distance(21:40,3),'filled','y');
% scatter3(Distance(41:60,1),Distance(41:60,2),Distance(41:60,3),'filled','b');
% scatter3(Distance(61:80,1),Distance(61:80,2),Distance(61:80,3),'filled','m');

scatter3(Distance(1:100,1),Distance(1:100,2),Distance(1:100,3),'filled','r');
scatter3(Distance(101:200,1),Distance(101:200,2),Distance(101:200,3),'filled','y');
scatter3(Distance(201:300,1),Distance(201:300,2),Distance(201:300,3),'filled','b');
% scatter3(Distance(301:400,1),Distance(301:400,2),Distance(301:400,3),'filled','m');

% scatter3(Distance(1:200,1),Distance(1:200,2),Distance(1:200,3),'filled','b');
% scatter3(Distance(201:400,1),Distance(201:400,2),Distance(201:400,3),'filled','r');
% scatter3(Distance(401:600,1),Distance(401:600,2),Distance(401:600,3),'filled','g');
% scatter3(Distance(601:800,1),Distance(601:800,2),Distance(601:800,3),'filled','m');
title('MDS analysis 3D')
xlabel('MDS dimension 1')
ylabel('MDS dimension 2')
zlabel('MDS dimension 3')
legend('Phi0.51Chi3.8', 'Phi0.54Chi3.8', 'Phi0.60Chi3.8')%,'GrainSize(80,80)')

%'GrainSize(10,10)', 'GrDainSize(10,80)', 'GrainSize(80,10)','GrainSize(80,80)'
%'GrainSize(20,40)', 'GrainSize(40,20)', 'GrainSize(40,40)','GrainSize(80,80)'