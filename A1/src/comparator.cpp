#include<bits/stdc++.h>

using namespace std;

int main(int argc, char **argv){

    if(argc !=3){
        cout << "Usage: ./main <input_file> <output_file>" << endl;
        return EXIT_FAILURE;
    }
    const string input_file1{ argv[1] };
    const string input_file2{ argv[2] };

    ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);
    //use freopen to read input file with ios base
    
    // read from input file with fstream
    fstream fin1(input_file1, ios::in);
    if(!fin1){
        cout << "Cannot open input file" << endl;
        return EXIT_FAILURE;
    }

    fstream fin2(input_file2, ios::in);
    if(!fin2){
        cout << "Cannot open input file" << endl;
        return EXIT_FAILURE;
    }

    int ln = 0;
    while(!fin1.eof() && !fin2.eof()){
        string line1;
        getline(fin1, line1);

        string line2;
        getline(fin2, line2);
        
        stringstream ss1(line1);
        stringstream ss2(line2);
        set<int> transaction1, transaction2;

        while(getline(ss1, line1, ' ')){
            int item = stoi(line1);
            transaction1.insert(item);
        }

        while(getline(ss2, line2, ' ')){
            transaction2.insert(stoi(line2));
        }

        if(transaction1 != transaction2){
            cout << "Not equal at line " << ln << endl;
            cout << "Transaction 1: ";
            for(auto u : transaction1){
                cout << u << " ";
            }
            cout << endl;
            cout << "Transaction 2: ";
            for(auto u : transaction2){
                cout << u << " ";
            }
            cout << endl;
            return EXIT_FAILURE;
        }
        ln++;
    }
    
    cout<<"Equal"<<endl;
    return EXIT_SUCCESS;
}