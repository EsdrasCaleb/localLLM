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

    @Test
    void testAddSubNode() {
        BrowseNode node = new BrowseNode();
        BrowseNode subNode = Mockito.mock(BrowseNode.class);
        node.addSubNode(subNode);
        assertTrue(node.getSubNodes().contains(subNode));
    }
}
