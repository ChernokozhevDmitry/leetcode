class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        std::vector<std::vector<int>> res{{1}};
        for(int i = 0; i < numRows - 1; ++i){
            res.push_back({});
            res.back().push_back(1);
            for(int j = 1; j <= i; ++j){
                res.back().push_back(res[i][j-1] + res[i][j]);
            }
            res.back().push_back(1);
        }
        return res;        
    }
};