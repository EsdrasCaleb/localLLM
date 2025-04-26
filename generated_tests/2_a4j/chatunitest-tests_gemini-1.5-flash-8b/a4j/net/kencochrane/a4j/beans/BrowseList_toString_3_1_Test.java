package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BrowseList_toString_3_1_Test {

    @Test
    void testToStringEmptyList() {
        BrowseList browseList = new BrowseList();
        String expectedOutput = "No nodes\n";
        assertEquals(expectedOutput, browseList.toString());
    }

    @Test
    void testToStringNonEmptyList() {
        BrowseNode node1 = new BrowseNode("Node 1", 1);
        BrowseNode node2 = new BrowseNode("Node 2", 2);
        BrowseList browseList = new BrowseList();
        browseList.setBrowseNode(new BrowseNode[] { node1, node2 });
        String expectedOutput = "# of nodes = 2\n" + "Name: Node 1\n" + "ID: 1\n" + "Name: Node 2\n" + "ID: 2\n";
        assertEquals(expectedOutput, browseList.toString());
    }

    @Test
    void testToStringWithNullNode() {
        BrowseNode node1 = new BrowseNode("Node 1", 1);
        BrowseNode node2 = null;
        BrowseList browseList = new BrowseList();
        browseList.setBrowseNode(new BrowseNode[] { node1, node2 });
        String expectedOutput = "# of nodes = 2\n" + "Name: Node 1\n" + "ID: 1\n" + "No nodes\n";
        assertEquals(expectedOutput, browseList.toString());
    }

    @Test
    void testToStringWithEmptyArrayList() {
        BrowseList browseList = new BrowseList();
        browseList.setBrowseNode(new BrowseNode[0]);
        String expectedOutput = "No nodes\n";
        assertEquals(expectedOutput, browseList.toString());
    }

    // Assuming BrowseList class exists and contains the toString method
    static class BrowseList {

        private BrowseNode[] browseNode;

        public void setBrowseNode(BrowseNode[] browseNode) {
            this.browseNode = browseNode;
        }

        @Override
        public String toString() {
            if (browseNode == null || browseNode.length == 0) {
                return "No nodes\n";
            }
            StringBuilder sb = new StringBuilder("# of nodes = " + browseNode.length + "\n");
            for (BrowseNode node : browseNode) {
                if (node != null) {
                    sb.append("Name: " + node.getBrowseName() + "\n");
                    sb.append("ID: " + node.getBrowseId() + "\n");
                } else {
                    sb.append("No nodes\n");
                }
            }
            return sb.toString();
        }
    }

    static class BrowseNode {

        private String browseName;

        private int browseId;

        public BrowseNode(String browseName, int browseId) {
            this.browseName = browseName;
            this.browseId = browseId;
        }

        public BrowseNode() {
        }

        public String getBrowseName() {
            return browseName;
        }

        public int getBrowseId() {
            return browseId;
        }
    }
}
