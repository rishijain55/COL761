#include<bits/stdc++.h>
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

    string line;
    map<string,int> vertex_lable;
    int label_num = 0;
    bool first = true;
    while(getline(cin, line)){
        if(line.empty()){
            if(first){
                break;
            }
            else{
                first = true;
                continue;
            }
        }
        if(first){
            cout<<"t # "<<line.substr(1,int(line.size())-1)<<"\n";
            first = false;
            continue;
        }
        int num_vertex = stoi(line);

        string label;
        for(int i=0;i<num_vertex;++i){
            getline(cin, label);
            if(vertex_lable.find(label)==vertex_lable.end())
                vertex_lable[label] = label_num++;
            cout<<"v "<<i<<" "<<vertex_lable[label]<<"\n";
        }

        getline(cin, line);
        int num_edges = stoi(line); 
        for(int i=0;i<num_edges;++i){
            getline(cin, line);
            stringstream ss(line);
            int u,v,l;
            ss>>u>>v>>l;
            cout<<"e "<<u<<" "<<v<<" "<<l<<"\n";
        }
    }

}
