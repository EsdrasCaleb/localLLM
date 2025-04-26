package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class BrowseNode_getSubNode_6_0_Test {

    @Mock
    private ArrayList subNodes;

    @InjectMocks
    private BrowseNode browseNode;

    @BeforeEach
    public void setUp() {
        browseNode.setSubNodes(subNodes);
    }

    @Test
    public void testGetSubNode_NodeExists() {
        BrowseNode subNode = new BrowseNode();
        subNode.setBrowseId("123");
        subNodes.add(subNode);
        BrowseNode result = browseNode.getSubNode("123");
        assertEquals(subNode, result);
    }

    @Test
    public void testGetSubNode_NodeDoesNotExist() {
        BrowseNode subNode = new BrowseNode();
        subNode.setBrowseId("123");
        subNodes.add(subNode);
        BrowseNode result = browseNode.getSubNode("456");
        assertNull(result);
    }

    @Test
    public void testGetSubNode_EmptySubNodes() {
        BrowseNode result = browseNode.getSubNode("123");
        assertNull(result);
    }
}
