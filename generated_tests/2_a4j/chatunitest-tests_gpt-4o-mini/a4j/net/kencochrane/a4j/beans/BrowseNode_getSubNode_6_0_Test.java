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

class BrowseNode_getSubNode_6_0_Test {

    private BrowseNode browseNode;

    @BeforeEach
    void setUp() {
        browseNode = new BrowseNode();
    }

    @Test
    void testGetSubNode_WithMatchingId_ReturnsNode() throws Exception {
        BrowseNode subNode = new BrowseNode();
        subNode.setBrowseId("testId");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(subNode);
        setPrivateField(browseNode, "subNodes", subNodes);
        BrowseNode result = browseNode.getSubNode("testId");
        assertNotNull(result);
        assertEquals("testId", result.getBrowseId());
    }

    @Test
    void testGetSubNode_WithMatchingId_CaseInsensitive_ReturnsNode() throws Exception {
        BrowseNode subNode = new BrowseNode();
        subNode.setBrowseId("TESTID");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(subNode);
        setPrivateField(browseNode, "subNodes", subNodes);
        BrowseNode result = browseNode.getSubNode("testid");
        assertNotNull(result);
        assertEquals("TESTID", result.getBrowseId());
    }

    @Test
    void testGetSubNode_WithNonMatchingId_ReturnsNull() throws Exception {
        BrowseNode subNode = new BrowseNode();
        subNode.setBrowseId("testId");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(subNode);
        setPrivateField(browseNode, "subNodes", subNodes);
        BrowseNode result = browseNode.getSubNode("nonExistentId");
        assertNull(result);
    }

    @Test
    void testGetSubNode_WithEmptySubNodes_ReturnsNull() throws Exception {
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        setPrivateField(browseNode, "subNodes", subNodes);
        BrowseNode result = browseNode.getSubNode("testId");
        assertNull(result);
    }

    @Test
    void testGetSubNode_WithNullSubNodes_ReturnsNull() throws Exception {
        setPrivateField(browseNode, "subNodes", null);
        BrowseNode result = browseNode.getSubNode("testId");
        assertNull(result);
    }

    private void setPrivateField(BrowseNode browseNode, String fieldName, Object value) throws Exception {
        Field field = BrowseNode.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(browseNode, value);
    }
}
