class Solution {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& mat) {
        std::vector<int> res;
        int row = 0, col = 0, rowsize = mat.size(), colsize = mat[0].size();
        bool up = true;
        for(int k = 0; k < colsize * rowsize; ++k){
            res.push_back(mat[row][col]);
            if (up){
                if ((row == 0)&&(col < colsize - 1)){
                    ++col;
                }
                else if (col == colsize - 1){
                    ++row;
                }
                else {
                    --row;
                    ++col;
                    continue;
                }
                up = !up;
            }
            else {
                if ((row < rowsize - 1)&&(col == 0)){
                    ++row;
                }
                else if (row == rowsize - 1){
                    ++col;
                }
                else {
                    ++row;
                    --col;
                    continue;
                }
                up = !up;
            }
        }
        return res;
    }
};