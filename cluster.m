function cluster(i)
    system(horzcat(['unset LD_LIBRARY_PATH; ','./cluster ',num2str(i)]));
end