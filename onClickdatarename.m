clc
clear all
close all


path =("C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles\images\3.8") ;
cd (path)
files = dir (strcat(path,'\*.PNG'));

L = length (files);

for i=1:L
    disp(files(i).name)
    name=files(i).name;
    pre=int2str(i);
    ext='.png';
    new=strcat(pre,ext);

    copyfile(name,new)
    

end
