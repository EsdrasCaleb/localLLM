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

class BrowseNode_addSubNode_5_1_Test {

    @Test
    void addSubNode() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        BrowseNode parentNode = new BrowseNode();
        BrowseNode childNode = new BrowseNode();
        // Important:  Initialize childNode's fields for a complete test case
        childNode.setBrowseId("childId");
        childNode.setBrowseName("childName");
        // Act
        parentNode.addSubNode(childNode);
        // Assert
        ArrayList subNodes = parentNode.getSubNodes();
        assertTrue(subNodes.contains(childNode));
        // Check if the size is correct
        assertEquals(1, subNodes.size());
        // Add a second child node
        BrowseNode childNode2 = new BrowseNode();
        childNode2.setBrowseId("childId2");
        childNode2.setBrowseName("childName2");
        parentNode.addSubNode(childNode2);
        assertEquals(2, subNodes.size());
        assertTrue(subNodes.contains(childNode2));
        // Test with empty list
        BrowseNode parentNode2 = new BrowseNode();
        parentNode2.addSubNode(null);
        assertEquals(0, parentNode2.getSubNodes().size());
    }
}
