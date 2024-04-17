class Solution {
public:
    std::vector<int> spiralOrder(std::vector<std::vector<int>>& matrix) {
        std::vector<int> res;
        int rowsize = matrix.size(),
            colsize = matrix[0].size(),
            up_row = 0,
            down_row =  rowsize - 1,
            left_col = 0,
            right_col = colsize - 1;  
        while(res.size() < rowsize * colsize ){
            for(int i = left_col; i <= right_col; ++i){
                res.push_back(matrix[up_row][i]);
            }
            for(int i = up_row + 1; i <= down_row; ++i){
                res.push_back(matrix[i][right_col]);
            }
            if (up_row != down_row) {
                for(int i = right_col - 1; i >= left_col; --i){
                res.push_back(matrix[down_row][i]);
                }
            }    
            if (left_col != right_col) {
                for(int i = down_row - 1; i > up_row; --i){
                    res.push_back(matrix[i][left_col]);
                }    
            }
            ++up_row;
            --down_row;
            ++left_col;
            --right_col;  
        }    
        return res;
    }
};