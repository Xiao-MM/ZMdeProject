package algorithm;

import java.util.Arrays;

public class MaximumGold {
    int[][] dirs = new int[][]{{-1,0}, {1,0}, {0,-1}, {0,1}};
    int[][] grid;
    int m,n;
    int maxGoldNum;
    int[][] isContained;
    /**
     *  1219. 黄金矿工
     * @param grid
     * @return
     */
    public int getMaximumGold(int[][] grid) {
        this.grid = grid;
        this.m = grid.length;
        this.n = grid[0].length;
        this.isContained = new int[m][n];
        for (int[] ints : isContained) {
            Arrays.fill(ints, 0);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isContained[i][j] == 1){
                    continue;
                }
                dfs(i, j, 0);
            }
        }
        return maxGoldNum;
    }

    // 每次以start作为起点遍历它可以走遍的所有元素
    public void dfs(int x, int y, int gold){
        // 表明该位置的数据已经被走过了无须再度枚举
        isContained[x][y] = 1;
        gold += grid[x][y];
        // 等会递归完需要恢复
        int temp = grid[x][y];
        maxGoldNum = Math.max(maxGoldNum, gold);
        for (int[] dir : dirs) {
            int i = x + dir[0];
            int j = y + dir[1];
            if (i >= 0 && i < m && j >= 0 && j < n && grid[i][j] != 0){
                // 避免其回头
                grid[x][y] = 0;
                dfs(i, j, gold);
            }
        }
        // 恢复置0的数据
        grid[x][y] = temp;
    }
}
