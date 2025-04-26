package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class BrowseNode_getSubNode_6_1_Test {

    @Test
    public void testGetSubNode() {
        BrowseNode browseNode = new BrowseNode();
        browseNode.setBrowseId("1");
        browseNode.setBrowseName("Test");
        browseNode.setMode("Mode");
        ArrayList subNodes = new ArrayList();
        subNodes.add(browseNode);
        browseNode.setSubNodes(subNodes);
        BrowseNode node = browseNode.getSubNode("1");
        assertEquals("1", node.getBrowseId());
    }
}
