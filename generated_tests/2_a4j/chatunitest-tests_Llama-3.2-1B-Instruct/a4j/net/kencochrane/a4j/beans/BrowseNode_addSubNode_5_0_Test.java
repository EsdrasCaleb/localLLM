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
public class BrowseNode_addSubNode_5_0_Test {

    @Mock
    private BrowseNode focalNode;

    @InjectMocks
    private BrowseNode focalNodeToTest;

    @Test
    public void testAddSubNode() {
        // Arrange
        // Create a new BrowseNode object
        BrowseNode newNode = new BrowseNode();
        newNode.setBrowseId("newBrowseId");
        newNode.setBrowseName("newBrowseName");
        newNode.setMode("newMode");
        // Act
        focalNode.addSubNode(newNode);
        // Assert
        // Verify that the new node is added to the focalNode
        assertEquals(newNode, focalNodeToTest.getSubNodes());
    }
}
