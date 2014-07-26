#alias python='C:/Python/2.7/python.exe'
alias xl='excel lap_results.xls'
alias tell='python collect_excel.py && xl'
alias run='python exec_all.py'
alias kill='python kill_screen.py'
alias temp='python temp.py'

up(){
    jarname=$2
    if [ "$2" == "v" ]
    then
        jarname=DeepBabel_verify
    fi
    cd scripts && python upload_jar.py $1 $jarname && cd ..
}
