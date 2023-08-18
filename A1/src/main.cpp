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


// void test_1()
// {
//     const Item a{ "A" };
//     const Item b{ "B" };
//     const Item c{ "C" };
//     const Item d{ "D" };
//     const Item e{ "E" };

//     const std::vector<Transaction> transactions{
//         { a, b },
//         { b, c, d },
//         { a, c, d, e },
//         { a, d, e },
//         { a, b, c },
//         { a, b, c, d },
//         { a },
//         { a, b, c },
//         { a, b, d },
//         { b, c, e }
//     };

//     const uint64_t minimum_support_threshold = 1;

//     const FPTree fptree{ transactions, minimum_support_threshold };

//     const std::set<Pattern> patterns = fptree_growth( fptree );
    
//     std::set<Pattern, mul_comp> patterns2;

//     for(auto u:patterns){
//         patterns2.insert(u);
//     }


//     for(auto u: patterns2){
//         for(auto v:u.first){
//             std::cout << v << " ";
//         }
//         std::cout << u.second*u.first.size() << std::endl;
//     }
// }

// void test_2()
// {
//     const Item a{ "A" };
//     const Item b{ "B" };
//     const Item c{ "C" };
//     const Item d{ "D" };
//     const Item e{ "E" };

//     const std::vector<Transaction> transactions{
//         { a, b, d, e },
//         { b, c, e },
//         { a, b, d, e },
//         { a, b, c, e },
//         { a, b, c, d, e },
//         { b, c, d },
//     };

//     const uint64_t minimum_support_threshold = 3;

//     const FPTree fptree{ transactions, minimum_support_threshold };

//     const std::set<Pattern> patterns = fptree_growth( fptree );

//     assert( patterns.size() == 19 );
//     assert( patterns.count( { { e, b }, 5 } ) );
//     assert( patterns.count( { { e }, 5 } ) );
//     assert( patterns.count( { { a, b, e }, 4 } ) );
//     assert( patterns.count( { { a, b }, 4 } ) );
//     assert( patterns.count( { { a, e }, 4 } ) );
//     assert( patterns.count( { { a }, 4 } ) );
//     assert( patterns.count( { { d, a, b }, 3 } ) );
//     assert( patterns.count( { { d, a }, 3 } ) );
//     assert( patterns.count( { { d, e, b, a }, 3 } ) );
//     assert( patterns.count( { { d, e, b }, 3 } ) );
//     assert( patterns.count( { { d, e, a }, 3 } ) );
//     assert( patterns.count( { { d, e }, 3 } ) );
//     assert( patterns.count( { { d, b }, 4 } ) );
//     assert( patterns.count( { { d }, 4 } ) );
//     assert( patterns.count( { { c, e, b }, 3 } ) );
//     assert( patterns.count( { { c, e }, 3 } ) );
//     assert( patterns.count( { { c, b }, 4 } ) );
//     assert( patterns.count( { { c }, 4 } ) );
//     assert( patterns.count( { { b }, 6 } ) );
// }

// void test_3()
// {
//     const Item a{ "A" };
//     const Item b{ "B" };
//     const Item c{ "C" };
//     const Item d{ "D" };
//     const Item e{ "E" };
//     const Item f{ "F" };
//     const Item g{ "G" };
//     const Item h{ "H" };
//     const Item i{ "I" };
//     const Item j{ "J" };
//     const Item k{ "K" };
//     const Item l{ "L" };
//     const Item m{ "M" };
//     const Item n{ "N" };
//     const Item o{ "O" };
//     const Item p{ "P" };
//     const Item s{ "S" };

//     const std::vector<Transaction> transactions{
//         { f, a, c, d, g, i, m, p },
//         { a, b, c, f, l, m, o },
//         { b, f, h, j, o },
//         { b, c, k, s, p },
//         { a, f, c, e, l, p, m, n }
//     };

//     const uint64_t minimum_support_threshold = 3;

//     const FPTree fptree{ transactions, minimum_support_threshold };

//     const std::set<Pattern> patterns = fptree_growth( fptree );

//     assert( patterns.size() == 18 );
//     assert( patterns.count( { { f }, 4 } ) );
//     assert( patterns.count( { { c, f }, 3 } ) );
//     assert( patterns.count( { { c }, 4 } ) );
//     assert( patterns.count( { { b }, 3 } ) );
//     assert( patterns.count( { { p, c }, 3 } ) );
//     assert( patterns.count( { { p }, 3 } ) );
//     assert( patterns.count( { { m, f, c }, 3 } ) );
//     assert( patterns.count( { { m, f }, 3 } ) );
//     assert( patterns.count( { { m, c }, 3 } ) );
//     assert( patterns.count( { { m }, 3 } ) );
//     assert( patterns.count( { { a, f, c, m }, 3 } ) );
//     assert( patterns.count( { { a, f, c }, 3 } ) );
//     assert( patterns.count( { { a, f, m }, 3 } ) );
//     assert( patterns.count( { { a, f }, 3 } ) );
//     assert( patterns.count( { { a, c, m }, 3 } ) );
//     assert( patterns.count( { { a, c }, 3 } ) );
//     assert( patterns.count( { { a, m }, 3 } ) );
//     assert( patterns.count( { { a }, 3 } ) );
// }

int main(int argc, const char *argv[])
{
    if(argc !=3){
        std::cout << "Usage: ./main <input_file> <output_file>" << std::endl;
        return EXIT_FAILURE;
    }
    const std::string input_file{ argv[1] };
    const std::string output_file{ argv[2] };

    ios_base::sync_with_stdio(false);cin.tie(NULL);cout.tie(NULL);
    //use freopen to read input file with ios base
    freopen(input_file.c_str(), "r", stdin);
    freopen(output_file.c_str(), "w", stdout);

    std::vector<vector<Transaction>> transactions;
    //read each line with space separated items
    int curind =-1;
    std::string line;
    int ln = 0;
    const int maxln = 20000;
    Item maxItem = 0;
    while(std::getline(std::cin, line)){
        if(ln%maxln==0){
            ln =0;
            transactions.push_back(vector<Transaction>());
            curind++;
        }
        std::istringstream iss(line);
        std::vector<Item> items;
        Item item;
        while(iss >> item){
            items.push_back(item);
            maxItem = max(maxItem, item);
        }
        transactions[curind].push_back(items);
        ln++;
    }

    // std::cout<<maxItem<<std::endl;

    std::vector<std::set<Item>> Pattern_Mapping;
    for(auto u:transactions){
        // cout<<"new transaction"<<endl;
        const uint64_t minimum_support_threshold = 1000;
        set<Pattern> patterns;
        const FPTree fptree{ u, minimum_support_threshold };
        patterns = fptree_growth( fptree );
        set<Pattern, mul_comp> patterns2;
        for(auto u:patterns){
            patterns2.insert(u);
        }
        for(auto u: patterns2){
            if(u.first.size()>1){
                Pattern_Mapping.push_back(u.first);
            }
            // for(auto v:u.first){
            //     std::cout << v << " ";
            // }
            // std::cout << u.second << std::endl;
        }

    }


    maxItem++;
    map<long long, vector<int>> replaced_transaction;
    for(int i =0; i<Pattern_Mapping.size(); i++){
        replaced_transaction[-i-1] = vector<int>();
    }
    vector<vector<int>> new_transactions;
    for(int i =0; i<transactions.size();i++){
        auto transaction_block = transactions[i];
        for(int j =0; j<transaction_block.size(); j++){
            auto transaction = transaction_block[j];
            map<long long, int> mp;

            for(auto item:transaction){
                mp[item]++;
            }
            for(int k =0; k<Pattern_Mapping.size(); k++){
                auto pattern = Pattern_Mapping[k];
                bool flag = true;
                for(auto item:pattern){
                    if(mp[item]==0){

                        flag = false;
                        break;
                    }
                }
                if(flag){
                    for(auto item:pattern){
                        mp[item]--;
                        // if(mp[item]==0){
                        //     mp.erase(item);
                        // }
                    }
                    mp[-k-1]++;
                    if(replaced_transaction[-k-1].size()<2){
                        int transaction_id = maxln*(i) + (j);
                        replaced_transaction[-k-1].push_back(transaction_id);
                    }
                }
            }
            vector<int> new_transaction;
            for(auto u:mp){
                for(int k =0; k<u.second; k++){
                    new_transaction.push_back(u.first);
                }
            }
            new_transactions.push_back(new_transaction);
            
        }
    }    

    for(auto u:replaced_transaction){
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
            for(auto item:Pattern_Mapping[-u.first-1]){
                replace.push_back(item);
            } 
            new_transactions[u.second[0]] = replace;
        }
    }

    cout<<new_transactions.size()<<endl;
    for(auto u:new_transactions){
        for(auto v:u){
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }   

    cout<<endl;

    // cout<<"Pattern Mapping"<<endl;

    for(int i =0; i<Pattern_Mapping.size(); i++){
        if(replaced_transaction[-i-1].size()<2){
            continue;
        }
        cout<<-i-1<<" ";
        for(auto v:Pattern_Mapping[i]){
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    //write to output file
    // for(auto u:transactions){
    //     for(auto v:u){
    //         std::cout << v << " ";
    //     }
    //     std::cout << std::endl;
    // }
    



    return EXIT_SUCCESS;
}

