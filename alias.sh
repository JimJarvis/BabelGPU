#alias python='C:/Python/2.7/python.exe'
alias xl='excel deep_results.xls &'
alias tell='python collect_excel.py && xl'
alias run='python exec_all.py'
alias kill='python kill_screen.py'
alias temp='python temp.py'
alias query='cat experiment.txt | grep'

up(){
    jarname=$2
    if [ "$2" == "v" ]
    then
        jarname=DeepBabel_verify
    fi
    cd scripts && python upload_jar.py $1 $jarname && cd ..
}
