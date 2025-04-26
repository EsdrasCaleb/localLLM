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

    private BrowseNode browseNode;

    @BeforeEach
    void setUp() {
        browseNode = new BrowseNode();
    }

    @Test
    void testGetSubNode_WhenSearchFails_ShouldReturnNull() {
        browseNode.setBrowseId("123");
        browseNode.setSubNodes(new ArrayList<>());
        browseNode.setMode("Normal");
        assertEquals(null, browseNode.getSubNode("123"));
    }

    @Test
    void testGetSubNode_WhenSearchMatches_ShouldReturnSubNode() {
        browseNode.setBrowseId("123");
        browseNode.setSubNodes(new ArrayList<>());
        browseNode.setMode("Normal");
        BrowseNode subNode = browseNode.getSubNode("123");
        assertNotNull(subNode);
        assertEquals("123", subNode.getBrowseId());
        assertEquals("Normal", subNode.getMode());
    }
}
