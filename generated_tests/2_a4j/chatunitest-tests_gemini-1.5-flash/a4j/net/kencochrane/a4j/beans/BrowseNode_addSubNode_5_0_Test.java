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

public class BrowseNode_addSubNode_5_0_Test {

    @Test
    void testAddSubNode() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        BrowseNode node1 = new BrowseNode();
        node1.setBrowseId("1");
        node1.setBrowseName("Node 1");
        BrowseNode node2 = new BrowseNode();
        node2.setBrowseId("2");
        node2.setBrowseName("Node 2");
        // Act
        node1.addSubNode(node2);
        // Assert
        Field subNodesField = BrowseNode.class.getDeclaredField("subNodes");
        subNodesField.setAccessible(true);
        ArrayList subNodes = (ArrayList) subNodesField.get(node1);
        assertEquals(1, subNodes.size());
        assertSame(node2, subNodes.get(0));
        // Test with null input
        assertThrows(NullPointerException.class, () -> node1.addSubNode(null));
        // Test adding multiple nodes.
        BrowseNode node3 = new BrowseNode();
        node3.setBrowseId("3");
        node3.setBrowseName("Node 3");
        node1.addSubNode(node3);
        subNodes = (ArrayList) subNodesField.get(node1);
        assertEquals(2, subNodes.size());
        assertSame(node3, subNodes.get(1));
    }
}
