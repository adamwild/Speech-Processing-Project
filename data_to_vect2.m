clear all;


cwd = pwd;
file_vectors = strcat(cwd, '\vectors');

directory = strcat(cwd, '\files');
cd(directory)

%Words to be used for processing
% words = [string('one'); string('two'); string('three'); string('four'); string('five'); 
%     string('six'); string('seven'); string('eight'); string('nine'); string('zero')];
words = [ string('three'); string('four'); string('seven');string('nine'); string('zero')];

%Number of files to be used per directory
max_files = 1000;
for val = 500:1500
    max_files = val;
    s = '';
    for i = 1:length(words)

        word = words(i);
        word_dir = char(strcat(directory, '\', word));   
        cd(word_dir);

        %Code for processing file 'word' here
        filenames = dir(pwd);
        len = size(filenames);
        len = len(1);
        if (len-2)>max_files
            len = max_files+2;
        end

        for j = 3:len
            filename = filenames(j).name;
            [wave_file,Fs] = audioread(filename);

            %TO MODIFY - Apply feature extraction here
            [a,g] = lpc(wave_file,10);

            %-----------------------------------------
            lenS = size(s);
            lenS = lenS(1);
            if lenS == 0
                for k = 1:size(a(:))
                    s = strcat(s, string(k), ',');
                end
                s = strcat(s, 'y\n');
            end
            for k = 1:size(a(:))
                s = strcat(s, string(a(k)), ',');
            end
            s = strcat(s, word, '\n');

        end

    end

    %Saving new table of vectors/features
    cd(file_vectors);
    filenames = dir(pwd);
    len = size(filenames);
    len = len(1)-1;
    new_name = string(len-1);
    new_size = length(new_name);
    while new_size<3
        new_name = strcat('0', new_name);
        new_size = new_size + 1;
    end
    s = s.split('\n');

    file_vector = char(strcat(new_name,'.csv'));
    fid = fopen(file_vector,'wt');
    for z = 1:length(s)
        word = s(z);
        fprintf(fid, '%s\n', word);
    end

    fclose(fid);
end
cd (cwd)