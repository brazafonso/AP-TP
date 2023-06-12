for file in $(ls)
do
    if [ "$file" != "script.sh" ];
    then
        mv $file above_$file
    fi
done