function cluster(i)
    system(horzcat(['unset LD_LIBRARY_PATH; ','./matcher ',num2str(i)]));
end