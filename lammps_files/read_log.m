function [varargout] = readlog(varargin,i)
% Read LAMMPS log files
% input is log file name with path
% output is a structure --> out
% out.Chead --> has heading of columns in each run
% out.data  --> has the data during each run in the form of string 
% Type str2num(out.data{i}) to get the numeric array
%
% Example
%       logdata = readlog('log.LAMMPS');
%
%  Author :  sarunkarthi@gmail.com
%            http://web.ics.purdue.edu/~asubrama/pages/Research_Main.htm
%            Arun K. Subramaniyan
%            School of Aeronautics and Astronautics
%            Purdue University, West Lafayette, IN - 47907, USA.

logfile = varargin;
try
    fid = fopen(logfile,'r');
catch
    error('Log file not found!');
end
n=0;
loop = 1;
while feof(fid) == 0
    %----------- To get first line of thermo output --------
    while feof(fid) == 0
        a = fgetl(fid);
        if length(a) > 4 && strcmp(a(1:4),'Time')
            n=n+1;
            break;
        end
        
    end
    
    
    %----------------------Get Data-------------------------
    id = 1; %row id...
    while feof(fid) == 0 && n==i
        a = fgetl(fid);
        if strcmp(a(1),'L')
            loop = loop + 1;
            break;
        else
            if strcmp(a(1),'W')
                id=id;
               % break;
            else    
           logdata(id,1:size(str2num(a),2)) = str2num(a);
            id = id+1;
            end
            end
    end
    %--------------------------------------------------------
 
end
fclose(fid);

%--------OUTPUT-------------------------------------------

out.data = logdata;

varargout{1} = out;
