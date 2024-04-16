class Solution {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& mat) {
        std::vector<int> res;
        int row = mat.size();
        int col = mat[0].size();
        int total = row*col;
        int k=0, i=0, j=0;
        bool up=true;
        while(k<total) {
            res.push_back(mat[i][j]);
            if(up) {
                if(i==0 && j<col-1) {
                   j++;
                   up=!up;
                }
                else if(j== col-1) {
                    i++;
                    up= !up;
                }
                else {
                    i--;
                    j++;
                }
            }
            else {
                if(i<row-1 && j==0) {
                    i++;
                    up=!up;
                }
                else if(i == row -1) {
                    j++;
                    up = !up;
                }
                else {
                    i++;
                    j--;
                }
            }
            k++;
        }
        return res;
    }
};