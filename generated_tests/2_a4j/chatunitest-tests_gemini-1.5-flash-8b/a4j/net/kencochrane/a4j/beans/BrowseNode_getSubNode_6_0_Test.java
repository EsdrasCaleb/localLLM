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

class BrowseNode_getSubNode_6_0_Test {

    @Test
    void getSubNode_matchingId_returnsNode() {
        // Arrange
        BrowseNode node1 = new BrowseNode();
        node1.setBrowseId("id1");
        node1.setBrowseName("name1");
        BrowseNode node2 = new BrowseNode();
        node2.setBrowseId("id2");
        node2.setBrowseName("name2");
        ArrayList<BrowseNode> subNodes = new ArrayList<>(Arrays.asList(node1, node2));
        BrowseNode target = new BrowseNode();
        target.setSubNodes(subNodes);
        // Act
        BrowseNode result = target.getSubNode("id1");
        // Assert
        assertEquals(node1, result);
    }

    @Test
    void getSubNode_noMatchingId_returnsNull() {
        // Arrange
        BrowseNode node1 = new BrowseNode();
        node1.setBrowseId("id1");
        node1.setBrowseName("name1");
        BrowseNode node2 = new BrowseNode();
        node2.setBrowseId("id2");
        node2.setBrowseName("name2");
        ArrayList<BrowseNode> subNodes = new ArrayList<>(Arrays.asList(node1, node2));
        BrowseNode target = new BrowseNode();
        target.setSubNodes(subNodes);
        // Act
        BrowseNode result = target.getSubNode("id3");
        // Assert
        assertNull(result);
    }

    @Test
    void getSubNode_emptySubNodes_returnsNull() {
        // Arrange
        BrowseNode target = new BrowseNode();
        // Act
        BrowseNode result = target.getSubNode("id1");
        // Assert
        assertNull(result);
    }

    @Test
    void getSubNode_nullSubNodes_returnsNull() {
        // Arrange
        BrowseNode target = new BrowseNode();
        target.setSubNodes(null);
        // Act
        BrowseNode result = target.getSubNode("id1");
        // Assert
        assertNull(result);
    }

    @Test
    void getSubNode_matchingIdCaseInsensitive_returnsNode() {
        // Arrange
        BrowseNode node1 = new BrowseNode();
        node1.setBrowseId("Id1");
        node1.setBrowseName("name1");
        ArrayList<BrowseNode> subNodes = new ArrayList<>(Arrays.asList(node1));
        BrowseNode target = new BrowseNode();
        target.setSubNodes(subNodes);
        // Act
        BrowseNode result = target.getSubNode("id1");
        // Assert
        assertEquals(node1, result);
    }
}
