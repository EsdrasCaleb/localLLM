package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseNode_addSubNode_5_0_Test {

    @Test
    public void testAddSubNode() {
        // Arrange
        BrowseNode browseNode = new BrowseNode();
        BrowseNode subNode = new BrowseNode();
        // Act
        browseNode.addSubNode(subNode);
        // Assert
        assertNotNull(browseNode.getSubNodes());
        assertEquals(1, browseNode.getSubNodes().size());
        assertEquals(subNode, browseNode.getSubNodes().get(0));
    }
}
