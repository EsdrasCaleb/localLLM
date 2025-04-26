package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_getSubNode_6_0_Test {

    @InjectMocks
    private BrowseNode browseNode;

    @Mock
    private ArrayList<BrowseNode> subNodes;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        browseNode.setSubNodes(subNodes);
    }

    @Test
    public void testGetSubNodeFound() {
        BrowseNode subNode1 = new BrowseNode();
        subNode1.setBrowseId("1");
        subNode1.setBrowseName("Node 1");
        BrowseNode subNode2 = new BrowseNode();
        subNode2.setBrowseId("2");
        subNode2.setBrowseName("Node 2");
        when(subNodes.size()).thenReturn(2);
        when(subNodes.get(0)).thenReturn(subNode1);
        when(subNodes.get(1)).thenReturn(subNode2);
        BrowseNode result = browseNode.getSubNode("1");
        assertNotNull(result);
        assertEquals("1", result.getBrowseId());
        assertEquals("Node 1", result.getBrowseName());
    }

    @Test
    public void testGetSubNodeNotFound() {
        BrowseNode subNode1 = new BrowseNode();
        subNode1.setBrowseId("1");
        subNode1.setBrowseName("Node 1");
        when(subNodes.size()).thenReturn(1);
        when(subNodes.get(0)).thenReturn(subNode1);
        BrowseNode result = browseNode.getSubNode("2");
        assertNull(result);
    }

    @Test
    public void testGetSubNodeEmptyList() {
        when(subNodes.size()).thenReturn(0);
        BrowseNode result = browseNode.getSubNode("1");
        assertNull(result);
    }
}
