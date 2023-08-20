#include <cassert>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>
#include<bits/stdc++.h>
#include "../include/fptree.hpp"

using namespace std;

struct mul_comp{
    bool operator()(const  Pattern &p1, const Pattern &p2) const{
        return p1.first.size()>=p2.first.size();
    };
};
/*
create a function which sorts the input transactions on the basis of their size in descending order
partition the data set into blocks and get patterns for each block. Store the transaction id for each pattern in a vector
*/

int calc(vector<Transaction> &transactions){
    map<int,int> mp;
    for(auto u: transactions){
        for(auto v:u){
            mp[v]++;
        }
    }
    vector<int> v;
    for( auto u: mp){
        v.push_back(u.second);
    }
    sort(v.begin(),v.end());
    //get percentile
    int n = (v.size()*v.size())/(v.size()+35);
    return v[n];

}

void get_patterns(vector<Transaction> &transactions,int part_len, vector<vector<set<Item>>> &pattern_block, vector<vector<int>> &index_block){

        vector<Transaction> pos_transactions;//remove negative for pattern generation
        for(int i =0; i<transactions.size(); i++){
            Transaction t;
            for(auto u: transactions[i]){
                if(u>=0){
                    t.push_back(u);
                }
            }
            if(t.size()==0) t.push_back(0);
            pos_transactions.push_back(t);
        }

    vector<pair<int,int>> indices;
    int n = pos_transactions.size();
    for(int i =0; i<n; i++){
        indices.push_back({pos_transactions[i].size(),i});
    }

    sort(indices.begin(), indices.end(), greater<pair<int,int>>());

    //generate patterns
    vector<vector<Transaction>> newTransactions;
    int cind = -1;
    for(int i =0 ;i<n;i++){
        if(i%part_len==0){
            cind++;
            newTransactions.push_back(vector<Transaction>());
        }
        // if(pos_transactions[indices[i].second].size()>0)
        newTransactions[cind].push_back(pos_transactions[indices[i].second]);

    }
    //put index into index block
    cind =-1;
    for(int i =0;i<n;i++){
        if(i%part_len==0){
            cind++;
            index_block.push_back(vector<int>());
        }
        index_block[cind].push_back(indices[i].second);
    }

    pattern_block.resize(cind+1);
    for(int i =0;i<=cind;i++){
        int sz = newTransactions[i].size();
        const uint64_t minimum_support_threshold =calc(newTransactions[i]);
        // cout<<i<<" "<<minimum_support_threshold<<endl;
        set<Pattern> patterns;
        const FPTree fptree{ newTransactions[i], minimum_support_threshold };
        patterns = fptree_growth( fptree );
        set<Pattern, mul_comp> patterns2;
        for(auto u:patterns){
            patterns2.insert(u);
        }
        int pat_sz = patterns2.size();
        pat_sz = min(pat_sz,part_len*10);
        int j =0;
        for(auto u:patterns2){
            if(u.first.size()>1){
                pattern_block[i].push_back(u.first);
                if(j++==pat_sz){
                    break;
                }
            }
        }
    }
}

void output_mappings(vector<set<Item>> &mappings, map<long long, vector<int>> &replaced_transaction, long long counter = 0){
    for(int i =0; i<mappings.size(); i++){
        if(replaced_transaction[-i-1].size()<2){
            continue;
        }
        cout<<-i+counter<<" ";
        for(auto v : mappings[i]){
            cout << v << " ";
        }
        cout << endl;
    }
    // cout<<endl;
}

void output_transactions(vector<Transaction> &transactions){


    cout<<endl;
    // cout<<"Transactions"<<endl;
    for(auto u: transactions){
        for(auto v:u){
            cout << v << " ";
        }
        cout << endl;
    }   

    cout<<endl;
    
}

void searchPatterns(vector<vector<set<Item>>> &patternsFound,vector<vector<int>> &transactionIds, vector<Transaction> &transactions, vector<Transaction> &new_transactions, map<long long, vector<int>> &replaced_transaction, vector<set<Item>> &patternMapping, long long & counter){
    // cout<<"inside search patterns"<<endl;
    new_transactions.resize(transactions.size());
    long long init_counter = counter;
    map<set<Item>,int> currFoundPattern;
    
    int num_blocks = patternsFound.size();
    // cout<<"num_blocks "<<num_blocks<<endl;
    for(int i=0;i<num_blocks;i++){
        for(auto tid : transactionIds[i]){
            map<long long, int> mp;
            auto& transaction = transactions[tid];
            for(auto item : transaction){
                mp[item]++;
            }
            int pat_n  = 10;
            for(int c_ind =0;c_ind<=pat_n;c_ind++){
                int ud =1;
                if(c_ind==0) ud = 0;
                for(int ud_in = 0;ud_in<=ud;ud_in++){
                    int pat_ind =i;
                    if(ud_in==0) pat_ind+=c_ind;
                    else pat_ind-=c_ind;
                    if(pat_ind>=0 && pat_ind<num_blocks) {
                        vector<set<Item>> & currPatterns = patternsFound[pat_ind];

                        for(auto & currPattern : currPatterns){
                            bool flag = true;
                            for(auto& item : currPattern){
                                if(mp[item]==0){
                                    flag = false;
                                    break;
                                }
                            }
                            if(flag){
                                for(auto item: currPattern){
                                    mp[item]--;
                                }
                                long long assigned_id;
                                
                                if(currFoundPattern.find(currPattern)==currFoundPattern.end()){
                                    assigned_id = counter;
                                    currFoundPattern[currPattern] = counter;
                                    patternMapping.push_back(currPattern);
                                    counter--;
                                }else{
                                    assigned_id = currFoundPattern[currPattern];
                                }

                                mp[assigned_id]++;
                                long long rid = assigned_id-init_counter-1;
                                if(replaced_transaction[rid].size()<2){
                                    int transaction_id = (tid);
                                    replaced_transaction[rid].push_back(transaction_id);
                                }
                            }


                            
                        }
                    }
                }
            }
            Transaction new_transaction;

            for(auto u:mp){
                for(int k =0; k<u.second; k++){
                    new_transaction.push_back(u.first);
                }
            }

            new_transactions[tid] = new_transaction;
        }
    }

    long long num_patterns = 0;
    for(auto u : replaced_transaction){
        if(u.second.size()>1){
            num_patterns++;
        }
        if(u.second.size()==1){
            Transaction replace = new_transactions[u.second[0]];
            
            for(int i =0; i<replace.size(); i++){
                int t_id = replace[i];
                if(t_id==u.first+init_counter+1){
                    swap(replace[i], replace[replace.size()-1]);
                    break;
                }
            }
            replace.pop_back();

            for(auto item : patternMapping[-u.first-1]){
                replace.push_back(item);
            } 

            new_transactions[u.second[0]] = replace;
        }
    }
}
    
int main(int argc, const char *argv[])
{
    if(argc !=3){
        cout << "Usage: ./main <input_file> <output_file>" << endl;
        return EXIT_FAILURE;
    }
    const string input_file{ argv[1] };
    const string output_file{ argv[2] };

    ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);
    //use freopen to read input file with ios base
    freopen(input_file.c_str(), "r", stdin);
    freopen(output_file.c_str(), "w", stdout);

    vector<Transaction> transactions;
    //read input file
    string line;
    while(getline(cin, line)){
        stringstream ss(line);
        string item;
        Transaction transaction;
        map<int,int> mp;
        while(getline(ss, item, ' ')){
            int x = stoi(item);
            if(mp.find(x)==mp.end()){
                mp[x] = 1;
                transaction.push_back(x);
            }
        }
        transactions.push_back(transaction);
    }
    //get patterns
    long long count = -1;
    int part_len = 500;
    vector<vector<set<Item>>> pattern_block;
    vector<vector<int>> index_block;
    get_patterns(transactions, part_len, pattern_block, index_block);
    
    vector<Transaction> new_transactions;
    map<long long, vector<int>> replaced_transaction;
    vector<set<Item>> patternMapping;
    long long prev_counter = count;

    searchPatterns(pattern_block, index_block, transactions, new_transactions, replaced_transaction, patternMapping, count);
    output_mappings(patternMapping, replaced_transaction, prev_counter);
    prev_counter = count;

    transactions.clear();
    int num_iter =10;
    while(num_iter--){
        vector<vector<set<Item>>> pattern_block2;
        vector<vector<int>> index_block2;
        get_patterns(new_transactions, part_len, pattern_block2, index_block2);
        vector<Transaction> new_transactions3;
        map<long long, vector<int>> replaced_transaction2;
        vector<set<Item>> patternMapping2;
        searchPatterns(pattern_block2, index_block2, new_transactions, new_transactions3, replaced_transaction2, patternMapping2, count);
        output_mappings(patternMapping2, replaced_transaction2, prev_counter);
        prev_counter = count;
;
        new_transactions = new_transactions3;
    }
    output_transactions(new_transactions);

    return EXIT_SUCCESS;
}
