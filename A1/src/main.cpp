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
        return p1.first.size()*p1.second>=p2.first.size()*p2.second;
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
    int n = (v.size()*1)/40;
    return n;

}

void get_patterns(vector<Transaction> &transactions,int part_len, vector<vector<set<Item>>> &pattern_block, vector<vector<int>> &index_block){
    vector<pair<int,int>> indices;
    int n = transactions.size();
    for(int i =0; i<n; i++){
        indices.push_back({transactions[i].size(),i});
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
        newTransactions[cind].push_back(transactions[indices[i].second]);

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
        cout<<i<<" "<<minimum_support_threshold<<endl;
        set<Pattern> patterns;
        const FPTree fptree{ newTransactions[i], minimum_support_threshold };
        patterns = fptree_growth( fptree );
        set<Pattern, mul_comp> patterns2;
        for(auto u:patterns){
            patterns2.insert(u);
        }
        int pat_sz = patterns2.size();
        pat_sz = min(pat_sz,part_len*5);
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

void output(vector<vector<int>> &transactions, vector<set<Item>> &mappings,map<long long, vector<int>> &replaced_transaction){
    // cout<<"Transactions"<<endl;
    for(auto u: transactions){
        for(auto v:u){
            cout << v << " ";
        }
        cout << endl;
    }   

    cout<<endl;
    // cout<<"Mappings"<<endl;
    for(int i =0; i<mappings.size(); i++){
        if(replaced_transaction[-i-1].size()<2){
            continue;
        }
        cout<<-i-1<<" ";
        for(auto v : mappings[i]){
            cout << v << " ";
        }
        cout << endl;
    }
}

void searchPatterns(vector<vector<set<Item>>> &patternsFound,vector<vector<int>> &transactionIds,
                    vector<Transaction> &transactions){
    // cout<<"inside search patterns"<<endl;

    map<long long, vector<int>> replaced_transaction;
    vector<vector<int>> new_transactions;
    
    map<set<Item>,int> currFoundPattern;
    vector<set<Item>> patternMapping;
    long long counter = -1;
    
    int num_blocks = patternsFound.size();
    // cout<<"num_blocks "<<num_blocks<<endl;
    for(int i=0;i<num_blocks;i++){
        vector<set<Item>>& currPatterns = patternsFound[i];
        // cout<<i<<" "<<currPatterns.size()<<endl;
        for(auto tid : transactionIds[i]){
            map<long long, int> mp;
            auto& transaction = transactions[tid];
            for(auto item : transaction){
                mp[item]++;
            }
            for(auto currPattern : currPatterns){
                bool flag = true;
                for(auto item : currPattern){
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

                    if(replaced_transaction[assigned_id].size()<2){
                        int transaction_id = (tid);
                        // cout<<-assigned_id<<endl;
                        replaced_transaction[assigned_id].push_back(transaction_id);
                    }
                }


                
            }
            vector<int> new_transaction;

            for(auto u:mp){
                for(int k =0; k<u.second; k++){
                    new_transaction.push_back(u.first);
                }
            }
            // for(auto u: new_transaction){
            //     cout<<u<<" ";
            // }
            // cout<<endl;
            new_transactions.push_back(new_transaction);
        }
    }


    for(auto u : replaced_transaction){
        if(u.second.size()==1){
            vector<int> replace = new_transactions[u.second[0]];
            
            for(int i =0; i<replace.size(); i++){
                int t_id = replace[i];
                if(t_id==u.first){
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

    output(new_transactions, patternMapping, replaced_transaction);
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
        while(getline(ss, item, ' ')){
            transaction.push_back(stoi(item));
        }
        transactions.push_back(transaction);
    }
    //get patterns
    int part_len = 5000;
    vector<vector<set<Item>>> pattern_block;
    vector<vector<int>> index_block;
    get_patterns(transactions, part_len, pattern_block, index_block);
    
    searchPatterns(pattern_block, index_block, transactions);


    return EXIT_SUCCESS;
}

