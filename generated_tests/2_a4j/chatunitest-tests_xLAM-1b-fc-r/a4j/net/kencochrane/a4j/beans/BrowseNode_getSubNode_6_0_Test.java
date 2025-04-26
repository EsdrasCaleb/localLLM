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

    @Test
    public void testGetSubNode() {
        BrowseNode node = new BrowseNode();
        node.setBrowseId("123");
        node.setBrowseName("Test Name");
        BrowseNode mockNode = mock(BrowseNode.class);
        when(mockNode.getBrowseId()).thenReturn("123");
        when(mockNode.getBrowseName()).thenReturn("Test Name");
        ArrayList<BrowseNode> subNodes = new ArrayList<>();
        subNodes.add(mockNode);
        node.setSubNodes(subNodes);
        BrowseNode result = node.getSubNode("123");
        assertEquals("Test Name", result.getBrowseName());
    }
}
