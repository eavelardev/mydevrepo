import java.util.*;

public class MySolution {
    public int largestIsland(int[][] grid) {
        
        Map<Integer, List<Integer>> graph = get_graph(grid);
        if (graph.size() == 0) return 1;
        Map<Integer, Boolean> visited = new HashMap<Integer, Boolean>();

        for (int node : graph.keySet()) 
            visited.put(node, false);
        
        int island = 0;
        Map<Integer, Set<Integer>> coasts = new HashMap<Integer, Set<Integer>>();
        List<Integer> islands_size = new ArrayList<Integer>();

        for (int node : graph.keySet()) 
            if (visited.get(node) == false) {
                islands_size.add(0);
                dfs(grid, graph, node, visited, coasts, island, islands_size);
                island += 1;
            }

        int size_largest_island = Collections.max(islands_size);
        Set<Set<Integer>> connections = new HashSet<Set<Integer>>(coasts.values());

        for (Set<Integer> connection : connections) {
            int sum_sizes = 1;

            for (int connected_island : connection) 
                sum_sizes += islands_size.get(connected_island);
            
            if (sum_sizes > size_largest_island)
                size_largest_island = sum_sizes;
        }

        return size_largest_island;        
    } 

    public static Map<Integer, List<Integer>> get_graph(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        Map<Integer, List<Integer>> graph = new HashMap<Integer, List<Integer>>();
        
        for (int pos = 0; pos < m*n; pos++)
            if (check_pos(grid, pos, 1)) {
                List<Integer> adjacents = new ArrayList<Integer>();
                for (int neighbor : get_neighbors(grid, pos))
                    if (check_pos(grid, neighbor, 1))
                        adjacents.add(neighbor);
 
                graph.put(pos, adjacents);
            } 

        return graph;
    }

    public static void dfs(int[][] grid, Map<Integer, List<Integer>> graph, int node, 
        Map<Integer, Boolean> visited, Map<Integer, Set<Integer>> coasts, 
            int island, List<Integer> islands_size) {
        
        visited.put(node, true);
        islands_size.set(island, islands_size.get(island) + 1);

        for (int neighbor : get_neighbors(grid, node)) 
            if (check_pos(grid, neighbor, 0)) {
                if (coasts.containsKey(neighbor))
                    coasts.get(neighbor).add(island);
                else
                    coasts.put(neighbor, new HashSet<Integer>(Arrays.asList(island)));
            }
        
        for (int next_node : graph.get(node))
            if (visited.get(next_node) == false) 
                dfs(grid, graph, next_node, visited, coasts, island, islands_size);
    }

    public static List<Integer> get_neighbors(int[][] grid, int pos) {
        int m = grid.length, n = grid[0].length;
        int[] coord = pos_to_coord(pos, n);
        int i = coord[0], j = coord[1];
        List<Integer> neighbors = new ArrayList<Integer>();

        for (int[] d: new int[][] {{i-1,j},{i,j-1},{i,j+1},{i+1,j}})
            if ((0 <= d[0] && d[0] < m) && (0 <= d[1] && d[1] < n))
                neighbors.add(coord_to_pos(new int[] {d[0], d[1]}, n));

        return neighbors;
    }

    public static int coord_to_pos(int[] coord, int n) {
        return n*coord[0] + coord[1];
    }
    
    public static int[] pos_to_coord(int pos, int n) {
        return new int[] {pos/n, pos%n};
    }

    public static Boolean check_pos(int[][] grid, int pos, int comp) {
        int[] coord = pos_to_coord(pos, grid[0].length);
        if (grid[coord[0]][coord[1]] == comp)
            return true;
        return false;
    }    
}
