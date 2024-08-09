function [LEAVESnet,MaxLevel] = HCAplus_GetLEAVESnet_Classification(HCAplusnet,MaxLevel)
NewMeans = [];
NewCL = [];
NewadaptiveSig = [];

if ~(HCAplusnet.numNodes == 0)
    for ChildIndex = 1:size(HCAplusnet.Child,2)
        if ~isempty(HCAplusnet.Child{1,ChildIndex})
            if HCAplusnet.Level > MaxLevel
                MaxLevel = HCAplusnet.Level;
            end
            [LEAVESnet_Child,MaxLevel] = HCAplus_GetLEAVESnet_Classification(HCAplusnet.Child{ChildIndex},MaxLevel);
            NewMeans = [NewMeans LEAVESnet_Child.Means];
            NewCL = [NewCL LEAVESnet_Child.CL];    
            NewadaptiveSig = [NewadaptiveSig LEAVESnet_Child.adaptiveSig];
        else
            NewMeans = [NewMeans HCAplusnet.Means(:,ChildIndex)];
            NewCL = [NewCL HCAplusnet.CL(:,ChildIndex)];
            NewadaptiveSig = [NewadaptiveSig HCAplusnet.adaptiveSig(:,ChildIndex)];
            if HCAplusnet.Level > MaxLevel
                MaxLevel = HCAplusnet.Level;
            end
        end
    end
end

LEAVESnet.Means = NewMeans;
LEAVESnet.CL = NewCL;
LEAVESnet.adaptiveSig = NewadaptiveSig;
LEAVESnet.numNodes = size(NewMeans,2);