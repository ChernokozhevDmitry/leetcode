class Solution {
public:
    std::vector<int> spiralOrder(std::vector<std::vector<int>>& matrix) {
        enum class Direction_Way {
            east = 1,
            south = 2,
            west = 3,
            north = 4,
        };
        std::vector<int> res;
        Direction_Way direction_way = Direction_Way::east;
        int count = 0,
            up_row = 0,
            down_row = matrix.size() - 1,
            left_col = 0,
            right_col = matrix[0].size() - 1;  
        for(int k = 0; k < matrix.size() * matrix[0].size(); ++k){
            switch(direction_way){
                case Direction_Way::east: {
                    res.push_back(matrix[up_row][count]);
                    if (count == right_col) {
                        count = ++up_row; 
                        direction_way = Direction_Way::south;
                    }
                    else {
                        ++count;
                    }
                    break;
                };
                case Direction_Way::south: {
                    res.push_back(matrix[count][right_col]);
                    if (count == down_row) {
                        count = --right_col; 
                        direction_way = Direction_Way::west;
                    }
                    else {
                        ++count;
                    }
                    break;
                };
                case Direction_Way::west: {
                    res.push_back(matrix[down_row][count]);
                    if (count == left_col) {
                        count = --down_row; 
                        direction_way = Direction_Way::north;
                    }
                    else {
                        --count;
                    }
                    break;
                };
                case Direction_Way::north: {
                    res.push_back(matrix[count][left_col]);
                    if (count == up_row) {
                        count = ++left_col; 
                        direction_way = Direction_Way::east;
                    }
                    else {
                        --count;
                    }
                    break;
                };
            }    
        }
        return res;
    }
};