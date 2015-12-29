%% ICP
pointcloud1 = dlmread('../TMPpointcloud1.txt',' ');
pointcloud2 = dlmread('../TMPpointcloud2.txt',' ');
[TR, TT, ER] = icp(pointcloud1',pointcloud2',20,'Matching','kDtree','SmartRejection',2);
dlmwrite('../TMPicpRt.txt', cat(1, cat(2, TR, TT), [0 0 0 1]), ' ');
% pc1 = pointCloud(pointcloud1);
% pc2 = pointCloud((TR*pointcloud2'+repmat(TT,1,size(pointcloud2',2)))');
% pcwrite(pc1,'test1','PLYFormat','binary');
% pcwrite(pc2,'test2','PLYFormat','binary');