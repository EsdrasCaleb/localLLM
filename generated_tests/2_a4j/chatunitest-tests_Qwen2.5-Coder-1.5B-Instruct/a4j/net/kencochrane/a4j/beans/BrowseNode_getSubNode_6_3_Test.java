package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class BrowseNode_getSubNode_6_3_Test {

    @Mock
    private BrowseNode mockSubNode1;

    @Mock
    private BrowseNode mockSubNode2;

    @InjectMocks
    private BrowseNode browseNode;

    @Test
    public void testGetSubNodeCaseInsensitive() {
        // Arrange
        when(mockSubNode1.getBrowseId()).thenReturn("Item1");
        when(mockSubNode2.getBrowseId()).thenReturn("item2");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(mockSubNode1);
        subNodes.add(mockSubNode2);
        browseNode.setSubNodes(subNodes);
        // Act
        BrowseNode result = browseNode.getSubNode("ITEM1");
        // Assert
        assertNotNull(result);
        assertEquals("Item1", result.getBrowseId());
    }

    @Test
    public void testGetSubNodeNotFound() {
        // Arrange
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        browseNode.setSubNodes(subNodes);
        // Act
        BrowseNode result = browseNode.getSubNode("Item3");
        // Assert
        assertNull(result);
    }
}
