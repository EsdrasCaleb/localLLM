package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_getSubNode_6_0_Test {

    @Test
    public void testGetSubNode() {
        // Arrange
        BrowseNode browseNode = new BrowseNode();
        browseNode.setBrowseId("123");
        browseNode.setBrowseName("Browse Node 1");
        browseNode.setMode("mode1");
        browseNode.setSubNodes(new ArrayList<>());
        browseNode.setParentNodes(new ArrayList<>());
        // Act
        BrowseNode result = browseNode.getSubNode("123");
        // Assert
        assertEquals(browseNode, result);
    }

    @Test
    public void testGetSubNode_EmptyList() {
        // Arrange
        BrowseNode browseNode = new BrowseNode();
        browseNode.setBrowseId("123");
        browseNode.setBrowseName("Browse Node 1");
        browseNode.setMode("mode1");
        browseNode.setSubNodes(new ArrayList<>());
        browseNode.setParentNodes(new ArrayList<>());
        // Act
        BrowseNode result = browseNode.getSubNode("123");
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetSubNode_NullBrowseId() {
        // Arrange
        BrowseNode browseNode = new BrowseNode();
        browseNode.setBrowseId("123");
        browseNode.setBrowseName("Browse Node 1");
        browseNode.setMode("mode1");
        browseNode.setSubNodes(new ArrayList<>());
        browseNode.setParentNodes(new ArrayList<>());
        // Act
        BrowseNode result = browseNode.getSubNode(null);
        // Assert
        assertNull(result);
    }
}
