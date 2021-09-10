read var<counter.txt
while true
do
    if [[ $(squeue | grep "collins") ]]; then
        clear -x;
        echo "currently running simulation $var";
        echo "$(squeue|grep "collins")"
    else
        let "var+=1"
        echo "$var">counter.txt
        echo "No simulation running... starting simulation $var";
        code=`sed "${var}q;d" ./missing.txt`
        eval "$code"
    fi
    sleep 60;
done
