package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BrowseNode_addSubNode_5_0_Test {

    @Mock
    private BrowseNode browseNode;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testAddSubNode() {
        BrowseNode subNode = new BrowseNode();
        when(browseNode.getSubNodes()).thenReturn(new ArrayList<>());
        browseNode.addSubNode(subNode);
        assertEquals(1, browseNode.getSubNodes().size());
        assertEquals(subNode, browseNode.getSubNodes().get(0));
    }
}
