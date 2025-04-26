package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_getSubNode_6_0_Test {

    @Test
    void testGetSubNode_found() throws Exception {
        BrowseNode node1 = new BrowseNode();
        node1.setBrowseId("123");
        node1.setBrowseName("Node 1");
        BrowseNode node2 = new BrowseNode();
        node2.setBrowseId("456");
        node2.setBrowseName("Node 2");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(node1);
        subNodes.add(node2);
        BrowseNode browseNode = new BrowseNode();
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(browseNode, subNodes);
        BrowseNode result = browseNode.getSubNode("123");
        assertEquals("123", result.getBrowseId());
        assertEquals("Node 1", result.getBrowseName());
    }

    @Test
    void testGetSubNode_notFound() throws Exception {
        BrowseNode node1 = new BrowseNode();
        node1.setBrowseId("123");
        node1.setBrowseName("Node 1");
        BrowseNode node2 = new BrowseNode();
        node2.setBrowseId("456");
        node2.setBrowseName("Node 2");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(node1);
        subNodes.add(node2);
        BrowseNode browseNode = new BrowseNode();
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(browseNode, subNodes);
        BrowseNode result = browseNode.getSubNode("789");
        assertNull(result);
    }

    @Test
    void testGetSubNode_emptySubNodes() throws Exception {
        BrowseNode browseNode = new BrowseNode();
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(browseNode, new ArrayList<>());
        BrowseNode result = browseNode.getSubNode("123");
        assertNull(result);
    }

    @Test
    void testGetSubNode_nullSubNodes() throws Exception {
        BrowseNode browseNode = new BrowseNode();
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(browseNode, null);
        BrowseNode result = browseNode.getSubNode("123");
        assertNull(result);
    }

    @Test
    void testGetSubNode_caseInsensitive() throws Exception {
        BrowseNode node1 = new BrowseNode();
        node1.setBrowseId("123");
        node1.setBrowseName("Node 1");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(node1);
        BrowseNode browseNode = new BrowseNode();
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        subNodesField.set(browseNode, subNodes);
        BrowseNode result = browseNode.getSubNode("   123   ");
        assertEquals("123", result.getBrowseId());
    }
}
