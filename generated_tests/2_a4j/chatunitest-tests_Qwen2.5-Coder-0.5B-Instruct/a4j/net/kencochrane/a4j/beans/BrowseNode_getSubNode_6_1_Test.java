package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BrowseNode_getSubNode_6_1_Test {

    @InjectMocks
    private BrowseNode browseNode;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetSubNodeWithValidId() {
        BrowseNode node = browseNode.getSubNode("123");
        assertNotNull(node);
        assertEquals("123", node.getBrowseId());
    }

    @Test
    public void testGetSubNodeWithInvalidId() {
        BrowseNode node = browseNode.getSubNode("abc");
        assertNull(node);
    }

    @Test
    public void testGetSubNodeWithNullList() {
        browseNode.setSubNodes(null);
        assertNull(browseNode.getSubNode("123"));
    }
}
