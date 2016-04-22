function timestr = getHMS(sec)

timestr = datestr(sec/24/3600, 'HH:MM:SS.FFF');
end