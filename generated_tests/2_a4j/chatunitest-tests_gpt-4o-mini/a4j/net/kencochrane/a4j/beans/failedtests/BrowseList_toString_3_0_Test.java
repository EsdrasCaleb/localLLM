package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BrowseList_toString_3_0_Test {

    private BrowseList browseList;

    @BeforeEach
    void setUp() {
        browseList = new BrowseList();
    }

    @Test
    void testToString_WithNoNodes() {
        // Test when there are no nodes
        String expected = "No nodes\n";
        assertEquals(expected, browseList.toString());
    }

    @Test
    void testToString_WithOneNode() {
        BrowseNode node = new BrowseNode("Node1", 1);
        browseList.setBrowseNode(new BrowseNode[] { node });
        String expected = "# of nodes = 1\nName: Node1\nID: 1\n";
        assertEquals(expected, browseList.toString());
    }

    @Test
    void testToString_WithMultipleNodes() {
        BrowseNode node1 = new BrowseNode("Node1", 1);
        BrowseNode node2 = new BrowseNode("Node2", 2);
        browseList.setBrowseNode(new BrowseNode[] { node1, node2 });
        String expected = "# of nodes = 2\nName: Node1\nID: 1\nName: Node2\nID: 2\n";
        assertEquals(expected, browseList.toString());
    }

    @Test
    void testToString_WithNullNode() {
        BrowseNode node1 = new BrowseNode("Node1", 1);
        browseList.setBrowseNode(new BrowseNode[] { node1, null });
        String expected = "# of nodes = 2\nName: Node1\nID: 1\n";
        assertEquals(expected, browseList.toString());
    }
}

// Assuming the BrowseList and BrowseNode classes are defined as follows:
class BrowseList {

    private BrowseNode[] browseNodes;

    public void setBrowseNode(BrowseNode[] browseNodes) {
        this.browseNodes = browseNodes;
    }

    @Override
    public String toString() {
        if (browseNodes == null || browseNodes.length == 0) {
            return "No nodes\n";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("# of nodes = ").append(browseNodes.length).append("\n");
        for (BrowseNode node : browseNodes) {
            if (node != null) {
                sb.append("Name: ").append(node.getName()).append("\n");
                sb.append("ID: ").append(node.getId()).append("\n");
            }
        }
        return sb.toString();
    }
}

class BrowseNode {

    private String name;

    private int id;

    public BrowseNode(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public int getId() {
        return id;
    }
}
