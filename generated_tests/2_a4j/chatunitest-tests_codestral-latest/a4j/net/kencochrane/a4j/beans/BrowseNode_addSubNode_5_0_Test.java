package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BrowseNode_addSubNode_5_0_Test {

    @InjectMocks
    private BrowseNode browseNode;

    @Mock
    private ArrayList<BrowseNode> subNodes;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        browseNode.setSubNodes(subNodes);
    }

    @Test
    void testAddSubNode() {
        BrowseNode subNode = mock(BrowseNode.class);
        browseNode.addSubNode(subNode);
        verify(subNodes).add(subNode);
    }
}
