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

public class BrowseNode_getSubNode_6_0_Test {

    private BrowseNode browseNode;

    private BrowseNode subNode1;

    private BrowseNode subNode2;

    @BeforeEach
    public void setUp() throws Exception {
        browseNode = new BrowseNode();
        subNode1 = new BrowseNode();
        subNode2 = new BrowseNode();
        subNode1.setBrowseId("123");
        subNode2.setBrowseId("456");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(subNode1);
        subNodes.add(subNode2);
        Field field = BrowseNode.class.getDeclaredField("subNodes");
        field.setAccessible(true);
        field.set(browseNode, subNodes);
    }

    @Test
    public void testGetSubNodeFound() {
        BrowseNode result = browseNode.getSubNode("123");
        assertNotNull(result);
        assertEquals("123", result.getBrowseId());
    }

    @Test
    public void testGetSubNodeNotFound() {
        BrowseNode result = browseNode.getSubNode("789");
        assertNull(result);
    }

    @Test
    public void testGetSubNodeIgnoreCase() {
        BrowseNode result = browseNode.getSubNode(" 123 ");
        assertNotNull(result);
        assertEquals("123", result.getBrowseId());
    }

    @Test
    public void testGetSubNodeEmptySubNodes() throws Exception {
        Field field = BrowseNode.class.getDeclaredField("subNodes");
        field.setAccessible(true);
        field.set(browseNode, new ArrayList<BrowseNode>());
        BrowseNode result = browseNode.getSubNode("123");
        assertNull(result);
    }

    @Test
    public void testGetSubNodeNullSubNodes() throws Exception {
        Field field = BrowseNode.class.getDeclaredField("subNodes");
        field.setAccessible(true);
        field.set(browseNode, null);
        BrowseNode result = browseNode.getSubNode("123");
        assertNull(result);
    }
}
