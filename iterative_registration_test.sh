for i in {1..20}; do alpha=$(echo "$i*0.31416" | bc); echo $alpha; echo $(./BinReg $alpha ${1} ${2}); printf '\n'; done
# use bc or awk for multiplication
