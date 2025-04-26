package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseNode_printNode_7_1_Test {

    @Test
    void testPrintNode() {
        BrowseNode browseNode = new BrowseNode();
        browseNode.setBrowseId("1");
        browseNode.setBrowseName("Node 1");
        browseNode.setMode("Mode 1");
        browseNode.setSubNodes(new ArrayList<>());
        browseNode.setParentNodes(new ArrayList<>());
        // Mock the parentNodes and subNodes
        ArrayList<BrowseNode> parentNodeMock = mock(ArrayList.class);
        when(parentNodeMock.get(0)).thenReturn(browseNode);
        when(parentNodeMock.size()).thenReturn(1);
        ArrayList<BrowseNode> subNodeMock = mock(ArrayList.class);
        when(subNodeMock.get(0)).thenReturn(new BrowseNode());
        when(subNodeMock.size()).thenReturn(1);
        browseNode.setParentNodes(parentNodeMock);
        browseNode.setSubNodes(subNodeMock);
        // Call the method under test
        browseNode.printNode();
    }
}
