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

    private BrowseNode browseNode;

    @BeforeEach
    public void setUp() {
        browseNode = Mockito.mock(BrowseNode.class);
    }

    @Test
    public void testAddSubNode() {
        BrowseNode sNode = Mockito.mock(BrowseNode.class);
        Mockito.when(browseNode.getParentNodes()).thenReturn(new ArrayList<>());
        Mockito.when(browseNode.getSubNodes()).thenReturn(new ArrayList<>());
        browseNode.addSubNode(sNode);
        assertEquals(1, browseNode.getParentNodes().size());
        assertEquals(1, browseNode.getSubNodes().size());
        assertEquals(sNode, browseNode.getParentNodes().get(0));
        assertEquals(sNode, browseNode.getSubNodes().get(1));
    }
}
