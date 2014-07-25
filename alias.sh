alias python='C:/Python/2.7/python.exe'
alias xl='excel lap_results.xls'
alias tell='python collect_excel.py && xl'
alias run='python exec_all.py'
alias kill='python kill_screen.py'
alias temp='python temp.py'

up(){
cd scripts && python upload_jar.py $1 $2 && cd ..
}
