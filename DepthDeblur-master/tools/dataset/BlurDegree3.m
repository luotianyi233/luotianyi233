clc;
clear;
%list of train sequence
sequence_name = {'GOPR0372_07_00'; 'GOPR0372_07_01'; 'GOPR0374_11_00';
    'GOPR0374_11_01'; 'GOPR0374_11_02'; 'GOPR0374_11_03'; 
    'GOPR0378_13_00'; 'GOPR0379_11_00'; 'GOPR0380_11_00';
    'GOPR0384_11_01'; 'GOPR0384_11_02'; 'GOPR0384_11_03';
    'GOPR0384_11_04'; 'GOPR0385_11_00'; 'GOPR0386_11_00';
    'GOPR0477_11_00'; 'GOPR0857_11_00'; 'GOPR0868_11_01';
    'GOPR0868_11_02'; 'GOPR0871_11_01'; 'GOPR0881_11_00'; 'GOPR0884_11_00'};
first_image = {323; 4207; 1; 2233; 5950; 27289; 22521; 2058; 1464; 10461; 
    14310; 23109; 30808; 1101; 2707; 1; 1; 2431; 7490; 1991; 1; 2036};
save_idx = {47; 602; 1; 204; 542; 2482; 41; 188; 134; 952; 1302; 2102; 2802; 101; 247; 1; 1; 222; 682; 182; 1; 186};
average_num = {7; 7; 11; 11; 11; 11; 13; 11; 11; 11; 11; 11; 11; 11; 11; 11; 11; 11; 11; 11; 11; 11};
image_num = {100; 74; 150; 79; 99; 47; 110; 100; 60; 99; 99; 99; 99; 100; 100; 80; 100; 99; 99; 99; 100; 100};
readpath = 'G:\GOPRO_Large_all\';
savepath = 'C:\Users\cjw-pc\Desktop\';

for idx = 1:1:size(sequence_name, 1)
    for j = 0:1:(image_num{idx} - 1)
        I = cell(1,average_num{idx});
        for i = 1:1:average_num{idx}
            name = num2str(first_image{idx} + j * average_num{idx} + i - 1, '%06d');
            temp = imread([readpath, sequence_name{idx}, '\',name, '.png']);
            I{i} = double(temp);
        end
        
        size_v = size(temp,1);
        size_u = size(temp,2);
        S = zeros(size_v,size_u);
        
        for v=1:1:size_v
            for u=1:1:size_u
                temp_color(:,1) = I{1}(v,u,:);
                for id_img=2:1:average_num{idx} % each img
                    temp_color(:,id_img) = I{id_img}(v,u,:);
                end
                S(v,u) = mean(std(temp_color, 0, 2));
            end
        end
        S = uint16(S * 100);
        savename = num2str(save_idx{idx} + j, '%06d');
        imwrite(S, [savepath, sequence_name{idx}, '\', savename, '.png']);
    end
end


% I = cell(1,11);
% for i = 1:1:11
%     name = num2str(1981 + i - 1, '%06d');
%     temp = imread([readpath, 'GOPR0410_11_00', '\',name, '.png']);
%     I{i} = double(temp);
% end
% size_v = size(temp,1);
% size_u = size(temp,2);
% S = zeros(size_v,size_u);
% for v=1:1:size_v
%     for u=1:1:size_u
%         temp_color(:,1) = I{1}(v,u,:);
%         for id_img=2:1:11 % each img
%             temp_color(:,id_img) = I{id_img}(v,u,:);
%         end
%         S(v,u) = mean(std(temp_color, 0, 2));
%     end
% end

% %show degree map
% flip=flipdim(S,1);
% h=pcolor(flip);%热度图
% set(h,'edgecolor','none');%去掉网格，平滑热度图
% axis equal
% colorbar;