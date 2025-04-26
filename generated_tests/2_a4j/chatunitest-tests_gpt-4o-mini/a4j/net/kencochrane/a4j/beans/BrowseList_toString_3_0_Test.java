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
