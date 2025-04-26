package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BrowseNode_getSubNode_6_0_Test {

    @Test
    void testGetSubNode() {
        // Arrange
        BrowseNode mockBrowseNode = Mockito.mock(BrowseNode.class);
        String browseId = "testId";
        Mockito.when(mockBrowseNode.getBrowseId()).thenReturn(browseId);
        Mockito.when(mockBrowseNode.getSubNodes()).thenReturn(new ArrayList<BrowseNode>());
        // Act
        BrowseNode result = mockBrowseNode.getSubNode(browseId);
        // Assert
        assertNotNull(result);
        assertEquals(browseId, result.getBrowseId());
    }
}
