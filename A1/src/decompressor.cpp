#include<bits/stdc++.h>
#include "../include/fptree.hpp"

using namespace std;

int main(int argc, char **argv){

    if(argc !=3){
        cout << "Usage: ./main <input_file> <output_file>" << endl;
        return EXIT_FAILURE;
    }

    const string input_file{ argv[1] };
    const string output_file{ argv[2] };
    cout<<"Input file: "<<input_file<<endl;
    cout<<"Output file: "<<output_file<<endl;

    ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);
    //use freopen to read input file with ios base
    freopen(input_file.c_str(), "r", stdin);
    freopen(output_file.c_str(), "w", stdout);

    map<long long, vector<int>> patterns;
    string line;

    while(getline(cin, line)){
        if(line.empty()){
            break;
        }

        stringstream ss(line);
        long long pattern_id;
        int item;
        bool first = true;
        while(ss >> item){
            if(first){
                first = false;
                pattern_id = item;
            }else{
                patterns[pattern_id].push_back(item);
            }
        }
    }

    vector<Transaction> transactions;

    while(getline(cin, line)){
        stringstream ss(line);
        Transaction transaction;
        while(getline(ss, line, ' ')){
            int item = stoi(line);
            if(item >= 0){
                transaction.push_back(item);
            }else{
                for(auto u : patterns[item]){
                    transaction.push_back(u);
                }
            }
        }

        transactions.push_back(transaction);
    }

    for(auto u : transactions){
        for(auto v : u){
            cout<<v<<" ";
        }
        cout<<endl;
    }

    return EXIT_SUCCESS;
}