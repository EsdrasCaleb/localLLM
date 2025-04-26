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
public class BrowseNode_getSubNode_6_1_Test {

    @InjectMocks
    private BrowseNode browseNode;

    @Test
    public void testGetSubNode_FindExistingSubNode() {
        // Arrange
        String browseId = "ExistingId";
        BrowseNode existingNode = new BrowseNode();
        existingNode.setBrowseId(browseId);
        existingNode.setBrowseName("ExistingName");
        existingNode.setMode("ExistingMode");
        browseNode.setSubNodes(new ArrayList<>(java.util.Arrays.asList(existingNode)));
        // Act
        BrowseNode foundNode = browseNode.getSubNode(browseId);
        // Assert
        assertEquals(existingNode, foundNode);
    }

    @Test
    public void testGetSubNode_FindNonExistingSubNode() {
        // Arrange
        String browseId = "NonExistingId";
        browseNode.setSubNodes(new ArrayList<>());
        // Act
        BrowseNode foundNode = browseNode.getSubNode(browseId);
        // Assert
        assertNull(foundNode);
    }

    @Test
    public void testGetSubNode_FindEmptySubNodes() {
        // Arrange
        String browseId = "ExistingId";
        browseNode.setSubNodes(new ArrayList<>());
        // Act
        BrowseNode foundNode = browseNode.getSubNode(browseId);
        // Assert
        assertNull(foundNode);
    }
}
