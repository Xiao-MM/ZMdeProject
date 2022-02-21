

import algorithm.CombinedSum2;
import algorithm.MaximumGold;
import algorithm.StaticMethod;
import algorithm.Subsets2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
//        System.out.println(StaticMethod.kthSmallest(new int[][]{
//                {1,5,9},
//                {10,11,13},
//                {12,13,15}
//        },8));
        System.out.println(StaticMethod.uniquePathsWithObstacles(new int[][]{
//                {0, 0, 0},
//                {0, 1, 0},
//                {0, 0, 0}
//                {0,1},{0,0}
                {1,0}
        }));

    }
}
