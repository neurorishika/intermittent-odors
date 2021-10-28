temp=$(squeue|grep "collins")
echo "${temp[0]}"
