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

class BrowseNode_addSubNode_5_0_Test {

    private BrowseNode parentNode;

    private BrowseNode childNode;

    @BeforeEach
    void setUp() {
        parentNode = new BrowseNode();
        childNode = new BrowseNode();
    }

    @Test
    void testAddSubNode() {
        // Verify that subNodes is initially empty
        assertTrue(parentNode.getSubNodes().isEmpty(), "SubNodes should be empty initially.");
        // Add a child node
        parentNode.addSubNode(childNode);
        // Verify that the child node has been added
        assertEquals(1, parentNode.getSubNodes().size(), "SubNodes should contain one element after adding a node.");
        assertSame(childNode, parentNode.getSubNodes().get(0), "The added subNode should be the same as the childNode.");
    }

    @Test
    void testAddMultipleSubNodes() {
        BrowseNode anotherChildNode = new BrowseNode();
        // Add multiple child nodes
        parentNode.addSubNode(childNode);
        parentNode.addSubNode(anotherChildNode);
        // Verify that both child nodes have been added
        assertEquals(2, parentNode.getSubNodes().size(), "SubNodes should contain two elements after adding two nodes.");
        assertSame(childNode, parentNode.getSubNodes().get(0), "The first subNode should be the same as the childNode.");
        assertSame(anotherChildNode, parentNode.getSubNodes().get(1), "The second subNode should be the same as anotherChildNode.");
    }

    @Test
    void testAddNullSubNode() {
        // Attempt to add a null child node
        parentNode.addSubNode(null);
        // Verify that null is added to subNodes
        assertEquals(1, parentNode.getSubNodes().size(), "SubNodes should contain one element after adding a null node.");
        assertNull(parentNode.getSubNodes().get(0), "The added subNode should be null.");
    }
}
