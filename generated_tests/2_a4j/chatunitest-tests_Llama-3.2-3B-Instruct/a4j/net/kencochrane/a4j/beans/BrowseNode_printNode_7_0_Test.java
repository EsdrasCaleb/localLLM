// BrowseNode_7_0_Test.java
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class BrowseNode_printNode_7_0_Test {

    @Test
    public void testPrintNode() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("testId");
        node.setBrowseName("testName");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(new BrowseNode());
        node.setSubNodes(subNodes);
        node.printNode();
        assertEquals("testId - testName", node.getBrowseId() + " - " + node.getBrowseName());
        assertEquals(1, node.getSubNodes().size());
        BrowseNode sNode = (BrowseNode) node.getSubNodes().get(0);
        assertEquals("testId - testName", sNode.getBrowseId() + " - " + sNode.getBrowseName());
    }

    @Test
    public void testPrintNode_EmptySubNodes() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("testId");
        node.setBrowseName("testName");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        node.setSubNodes(subNodes);
        node.printNode();
        assertEquals("testId - testName", node.getBrowseId() + " - " + node.getBrowseName());
        assertEquals(0, node.getSubNodes().size());
    }

    @Test
    public void testPrintNode_NullNode() {
        BrowseNode node = null;
        assertThrows(NullPointerException.class, () -> node.printNode());
    }
}
