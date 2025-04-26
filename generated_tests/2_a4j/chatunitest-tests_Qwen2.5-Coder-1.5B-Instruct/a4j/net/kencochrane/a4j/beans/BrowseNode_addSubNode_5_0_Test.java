package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BrowseNode_addSubNode_5_0_Test {

    @Test
    public void testAddSubNode() throws Exception {
        // Create an instance of BrowseNode
        BrowseNode rootNode = new BrowseNode();
        // Create a mock instance of BrowseNode
        BrowseNode childNode = mock(BrowseNode.class);
        // Set up expectations
        when(childNode.getBrowseId()).thenReturn("child1");
        when(childNode.getBrowseName()).thenReturn("Child Node");
        // Call the method to be tested
        rootNode.addSubNode(childNode);
        // Verify the state of the root node
        assertEquals(1, rootNode.getSubNodes().size());
        assertTrue(rootNode.getSubNodes().contains(childNode));
    }
}
